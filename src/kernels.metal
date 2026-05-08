#include <metal_stdlib>
using namespace metal;

// ── RMS Norm ──────────────────────────────────────────────────────────────────

/**
 * Root Mean Square Layer Normalization (RMSNorm).
 * 
 * Normalizes the input vector by its root mean square, then scales by weights.
 * Used to stabilize training and inference by maintaining activation variance.
 * 
 * @param x_out  Output buffer [dim]
 * @param x_in   Input buffer [dim]
 * @param w      Weight buffer for scaling [dim]
 * @param eps    Small epsilon value to prevent division by zero
 * @param dim    Dimensionality of the input/output vectors
 * @param tpig   Thread position in grid (only thread 0 executes)
 */
kernel void rms_norm(
    device float* x_out [[buffer(0)]],
    device const float* x_in [[buffer(1)]],
    device const float* w [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint tpig [[thread_position_in_grid]]
) {
    if (tpig >= 1) return; // Only one thread for now to simplify, handles whole row

    float ss = 0.0f;
    for (uint i = 0; i < dim; i++) {
        ss += x_in[i] * x_in[i];
    }
    float scale = 1.0f / sqrt(ss / (float)dim + eps);
    for (uint i = 0; i < dim; i++) {
        x_out[i] = x_in[i] * scale * w[i];
    }
}

// ── Rotary Position Embedding (RoPE) ──────────────────────────────────────────

/**
 * Rotary Position Embedding (RoPE).
 * 
 * Injects positional information into query and key vectors by rotating pairs 
 * of coordinates based on their position in the sequence.
 * 
 * @param q          Query buffer to rotate in-place
 * @param k          Key buffer to rotate in-place
 * @param pos        Current sequence position
 * @param head_dim   Dimensionality of each attention head
 * @param freq_base  Base frequency for the rotation angles (e.g., 10000.0)
 * @param tpig       Thread position in grid (handles pairs of dimensions)
 */
kernel void rope(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant uint& pos [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant float& freq_base [[buffer(4)]],
    uint tpig [[thread_position_in_grid]]
) {
    // tpig is the index into the heads (head_idx * head_dim + dim_idx)
    // We only process the first half of head_dim per thread to handle pairs
    uint i = tpig * 2;
    uint h_dim_idx = i % head_dim;

    float theta = pow(freq_base, -(float)h_dim_idx / (float)head_dim);
    float m_theta = (float)pos * theta;
    float cos_mt = cos(m_theta);
    float sin_mt = sin(m_theta);

    // Q rotation
    float q0 = q[i];
    float q1 = q[i+1];
    q[i]   = q0 * cos_mt - q1 * sin_mt;
    q[i+1] = q0 * sin_mt + q1 * cos_mt;

    // K rotation (assuming same logic)
    float k0 = k[i];
    float k1 = k[i+1];
    k[i]   = k0 * cos_mt - k1 * sin_mt;
    k[i+1] = k0 * sin_mt + k1 * cos_mt;
}

// ── Matrix-Vector Multiply ────────────────────────────────────────────────────

/**
 * Standard Matrix-Vector Multiplication (FP32).
 * 
 * Computes `out = W * x` where W is a matrix and x is a vector.
 * 
 * @param out   Output vector buffer [rows]
 * @param w     Weight matrix buffer [rows, cols] (row-major)
 * @param x     Input vector buffer [cols]
 * @param rows  Number of rows in W (output size)
 * @param cols  Number of columns in W (input size)
 * @param tpig  Thread position in grid (maps to the row index)
 */
kernel void matvec_f32(
    device float* out [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* x [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint tpig [[thread_position_in_grid]]
) {
    if (tpig >= rows) return;

    float sum = 0.0f;
    for (uint c = 0; c < cols; c++) {
        sum += w[tpig * cols + c] * x[c];
    }
    out[tpig] = sum;
}

/**
 * Batched Matrix-Vector Multiplication (FP32).
 * 
 * Computes `out[b] = W * x[b]` for multiple vectors in a batch.
 * 
 * @param out   Output matrix buffer [batch, rows]
 * @param w     Weight matrix buffer [rows, cols] (row-major)
 * @param x     Input matrix buffer [batch, cols]
 * @param rows  Number of rows in W (output size)
 * @param cols  Number of columns in W (input size)
 * @param tpig  Thread position in grid: x=row, y=batch_idx
 */
kernel void matvec_batched_f32(
    device float* out [[buffer(0)]], // [batch, rows]
    device const float* w [[buffer(1)]], // [rows, cols]
    device const float* x [[buffer(2)]], // [batch, cols]
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 tpig [[thread_position_in_grid]] // [row, batch_idx]
) {
    uint row = tpig.x;
    uint batch_idx = tpig.y;
    if (row >= rows) return;

    float sum = 0.0f;
    for (uint c = 0; c < cols; c++) {
        sum += w[row * cols + c] * x[batch_idx * cols + c];
    }
    out[batch_idx * rows + row] = sum;
}

// ── Matrix-Vector Multiply (Q4_0) ─────────────────────────────────────────────
// Block: 2 bytes scale (f16) + 16 bytes nibbles (32 elements) = 18 bytes.

/**
 * 4-bit Quantized Matrix-Vector Multiplication (Q4_0).
 * 
 * Computes `out = W * x` where W is quantized to 4-bit integers with a shared
 * FP16 scale factor per block of 32 elements.
 * 
 * @param out   Output vector buffer [rows]
 * @param w     Quantized weight matrix buffer
 * @param x     Input vector buffer [cols]
 * @param rows  Number of rows in W
 * @param cols  Number of columns in W
 * @param tpig  Thread position in grid (maps to row index)
 */
kernel void matvec_q4_0(
    device float* out [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const float* x [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint tpig [[thread_position_in_grid]]
) {
    if (tpig >= rows) return;

    float sum = 0.0f;
    uint n_blocks = cols / 32;
    uint row_offset = tpig * n_blocks * 18;

    for (uint b = 0; b < n_blocks; b++) {
        uint block_start = row_offset + b * 18;
        half delta = *(device const half*)(w + block_start);
        device const uchar* nibbles = w + block_start + 2;

        for (uint i = 0; i < 16; i++) {
            uchar byte = nibbles[i];
            float lo = (float)((int)(byte & 0x0F) - 8);
            float hi = (float)((int)(byte >> 4) - 8);
            sum += (float)delta * lo * x[b * 32 + i * 2];
            sum += (float)delta * hi * x[b * 32 + i * 2 + 1];
        }
    }
    out[tpig] = sum;
}

/**
 * Batched 4-bit Quantized Matrix-Vector Multiplication (Q4_0).
 * 
 * Computes `out[b] = W * x[b]` for a batch of vectors.
 * 
 * @param out   Output matrix buffer [batch, rows]
 * @param w     Quantized weight matrix buffer
 * @param x     Input matrix buffer [batch, cols]
 * @param rows  Number of rows in W
 * @param cols  Number of columns in W
 * @param tpig  Thread position in grid: x=row, y=batch_idx
 */
kernel void matvec_batched_q4_0(
    device float* out [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const float* x [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 tpig [[thread_position_in_grid]] // [row, batch_idx]
) {
    uint row = tpig.x;
    uint batch_idx = tpig.y;
    if (row >= rows) return;

    float sum = 0.0f;
    uint n_blocks = cols / 32;
    uint row_offset = row * n_blocks * 18;

    for (uint b = 0; b < n_blocks; b++) {
        uint block_start = row_offset + b * 18;
        half delta = *(device const half*)(w + block_start);
        device const uchar* nibbles = w + block_start + 2;

        for (uint i = 0; i < 16; i++) {
            uchar byte = nibbles[i];
            float lo = (float)((int)(byte & 0x0F) - 8);
            float hi = (float)((int)(byte >> 4) - 8);
            sum += (float)delta * lo * x[batch_idx * cols + b * 32 + i * 2];
            sum += (float)delta * hi * x[batch_idx * cols + b * 32 + i * 2 + 1];
        }
    }
    out[batch_idx * rows + row] = sum;
}

// ── Activation (SiLU) ─────────────────────────────────────────────────────────

/**
 * Sigmoid Linear Unit (SiLU) Activation.
 * 
 * Applies the SiLU activation function element-wise: `f(x) = x / (1 + exp(-x))`.
 * 
 * @param x     Input/Output buffer (modified in-place)
 * @param tpig  Thread position in grid (maps to vector element index)
 */
kernel void silu(
    device float* x [[buffer(0)]],
    uint tpig [[thread_position_in_grid]]
) {
    float val = x[tpig];
    x[tpig] = val / (1.0f + exp(-val));
}

// ── Element-wise Multiplication ──────────────────────────────────────────────

/**
 * Element-wise Vector Multiplication.
 * 
 * Computes `x[i] = x[i] * y[i]` for two vectors.
 * 
 * @param x     First vector buffer (modified in-place)
 * @param y     Second vector buffer (read-only)
 * @param tpig  Thread position in grid (maps to vector element index)
 */
kernel void vec_mul(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    uint tpig [[thread_position_in_grid]]
) {
    x[tpig] *= y[tpig];
}

// ── LoRA Kernels ─────────────────────────────────────────────────────────────

/**
 * Batched LoRA Matrix-Vector Multiplication.
 * 
 * Computes the LoRA correction delta for a batch of sequences where each 
 * sequence can have a different adapter. 
 * Math: out = out + (scale * (x * A) * B)
 * 
 * @param out           Output buffer to accumulate the LoRA delta [batch_size, d_out]
 * @param x             Input activation buffer [batch_size, d_in]
 * @param all_lora_a    Global pool of LoRA-A weights
 * @param all_lora_b    Global pool of LoRA-B weights
 * @param lora_offsets  Array of (offset_a, offset_b) pairs per batch row
 * @param lora_scales   Array of scaling factors per batch row
 * @param d_in          Input dimension size
 * @param d_out         Output dimension size
 * @param r             LoRA rank
 * @param batch_size    Number of sequences in the batch
 */
kernel void matvec_lora_batched(
    device float* out [[buffer(0)]],            // [batch_size, d_out] (Accumulates!)
    device const float* x [[buffer(1)]],        // [batch_size, d_in]
    device const float* all_lora_a [[buffer(2)]],
    device const float* all_lora_b [[buffer(3)]],
    device const uint2* lora_offsets [[buffer(4)]], // (offset_a, offset_b) per batch row
    device const float* lora_scales [[buffer(5)]],
    constant uint& d_in [[buffer(6)]],
    constant uint& d_out [[buffer(7)]],
    constant uint& r [[buffer(8)]],
    constant uint& batch_size [[buffer(9)]],
    uint2 tpig [[thread_position_in_grid]] // [col, batch_idx]
) {
    uint col = tpig.x;
    uint batch_idx = tpig.y;
    if (col >= d_out || batch_idx >= batch_size) return;

    uint2 offsets = lora_offsets[batch_idx];
    if (offsets.x == 0xFFFFFFFF) return; // No adapter for this row

    device const float* lora_a = all_lora_a + offsets.x;
    device const float* lora_b = all_lora_b + offsets.y;
    float scale = lora_scales[batch_idx];
    device const float* x_row = x + batch_idx * d_in;

    float lora_sum = 0.0f;
    for (uint j = 0; j < r; j++) {
        float xa = 0.0f;
        for (uint i = 0; i < d_in; i++) {
            xa += x_row[i] * lora_a[j * d_in + i];
        }
        lora_sum += xa * lora_b[col * r + j];
    }

    out[batch_idx * d_out + col] += lora_sum * scale;
}

// ── Softmax ───────────────────────────────────────────────────────────────────

/**
 * Softmax Activation.
 * 
 * Applies the Softmax function over a vector to convert logits into probabilities.
 * Stable implementation subtracts the maximum value before exponentiation.
 * 
 * @param x     Input/Output buffer (modified in-place)
 * @param n     Length of the vector
 * @param tpig  Thread position in grid (only thread 0 executes, processes full vector)
 */
kernel void softmax(
    device float* x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint tpig [[thread_position_in_grid]]
) {
    if (tpig >= 1) return; // One thread handles the whole row for now

    float max_val = -INFINITY;
    for (uint i = 0; i < n; i++) {
        max_val = max(max_val, x[i]);
    }

    float sum = 0.0f;
    for (uint i = 0; i < n; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }

    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

// ── Attention Kernels ─────────────────────────────────────────────────────────

/**
 * Flash Attention 2 for Paged KV Cache (FP32).
 * 
 * Implements the online softmax algorithm with one-pass tiling to minimize 
 * memory round-trips. O(1) memory overhead per head.
 * 
 * Supports Grouped Query Attention (GQA) by mapping multiple query heads
 * to a single KV head using: kv_head_idx = head_idx / (n_head / n_head_kv).
 * 
 * @param out           Output buffer for attention results [n_head, head_dim]
 * @param q             Query buffer for the current token [n_head, head_dim]
 * @param k_cache       Global KV cache pool for keys
 * @param v_cache       Global KV cache pool for values
 * @param head_dim      Dimensionality of each attention head
 * @param n_ctx         Total context length (not used in tiling loop)
 * @param pos           Current position of the token being generated
 * @param slot_id       ID of the active inference slot
 * @param slot_mapping  Buffer mapping slots to physical KV pages
 * @param max_pages     Total number of pages in the pool
 * @param block_size    Number of tokens per KV page
 * @param n_head_kv     Number of KV heads (for GQA)
 * @param n_head        Number of query heads
 */
kernel void paged_attention_flash(
    device float* out [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k_cache [[buffer(2)]],
    device const float* v_cache [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& n_ctx [[buffer(5)]],
    constant uint& pos [[buffer(6)]],
    constant uint& slot_id [[buffer(7)]],
    device const uint* slot_mapping [[buffer(8)]],
    constant uint& max_pages [[buffer(9)]],
    constant uint& block_size [[buffer(10)]],
    constant uint& n_head_kv [[buffer(11)]],
    constant uint& n_head [[buffer(12)]],
    uint head_idx [[thread_position_in_grid]]
) {
    uint kv_head_idx = head_idx / (n_head / n_head_kv);
    device const float* head_q = q + head_idx * head_dim;
    device float* head_out = out + head_idx * head_dim;

    float inv_sqrt_head_dim = 1.0f / sqrt((float)head_dim);
    
    float m = -INFINITY;
    float l = 0.0f;
    float acc[128]; // Supporting head_dim up to 128
    for (uint i = 0; i < head_dim; i++) acc[i] = 0.0f;

    uint total_tokens = pos + 1;

    // Flash Attention 2: One-pass tiling with online softmax
    for (uint t = 0; t < total_tokens; t++) {
        // 1. Fetch Key
        uint block_idx = slot_mapping[slot_id * max_pages + (t / block_size)];
        uint token_in_block = t % block_size;
        device const float* head_k = k_cache + (block_idx * block_size * n_head_kv * head_dim) + (token_in_block * n_head_kv * head_dim) + (kv_head_idx * head_dim);
        
        // 2. Compute Score
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += head_q[d] * head_k[d];
        }
        score *= inv_sqrt_head_dim;

        // 3. Online Softmax update
        float m_old = m;
        m = max(m_old, score);
        float exp_score = exp(score - m);
        float exp_m_diff = exp(m_old - m);
        l = l * exp_m_diff + exp_score;

        // 4. Accumulate Weighted Value
        device const float* head_v = v_cache + (block_idx * block_size * n_head_kv * head_dim) + (token_in_block * n_head_kv * head_dim) + (kv_head_idx * head_dim);
        for (uint d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * exp_m_diff + exp_score * head_v[d];
        }
    }

    // 5. Finalize output
    for (uint d = 0; d < head_dim; d++) {
        head_out[d] = acc[d] / l;
    }
}

/**
 * Tree-Based Paged Attention.
 * 
 * Verifies multiple speculative branches in parallel by following ancestor paths.
 * Each query at tree node 'i' only attends to tokens in its direct lineage.
 * 
 * @param ancestors_map  Buffer mapping each node to its ancestor indices in the tree.
 *                       Layout: [tree_size, max_depth] where -1 indicates no more ancestors.
 */
kernel void paged_attention_tree(
    device float* out [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k_cache [[buffer(2)]],
    device const float* v_cache [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& n_ctx [[buffer(5)]], // Prompt length
    constant uint& tree_size [[buffer(6)]],
    constant uint& slot_id [[buffer(7)]],
    device const uint* slot_mapping [[buffer(8)]],
    constant uint& max_pages [[buffer(9)]],
    constant uint& block_size [[buffer(10)]],
    constant uint& n_head_kv [[buffer(11)]],
    constant uint& n_head [[buffer(12)]],
    device const int* ancestors_map [[buffer(13)]],
    constant uint& max_tree_depth [[buffer(14)]],
    uint2 tg_pos [[thread_position_in_grid]] // [node_idx, head_idx]
) {
    uint node_idx = tg_pos.x;
    uint head_idx = tg_pos.y;
    if (node_idx >= tree_size || head_idx >= n_head) return;

    uint kv_head_idx = head_idx / (n_head / n_head_kv);
    device const float* node_q = q + (node_idx * n_head * head_dim) + (head_idx * head_dim);
    device float* node_out = out + (node_idx * n_head * head_dim) + (head_idx * head_dim);

    float inv_sqrt_head_dim = 1.0f / sqrt((float)head_dim);
    
    float m = -INFINITY;
    float l = 0.0f;
    float acc[128];
    for (uint i = 0; i < head_dim; i++) acc[i] = 0.0f;

    // 1. Attend to Prompt (already in KV cache)
    for (uint t = 0; t < n_ctx; t++) {
        uint block_idx = slot_mapping[slot_id * max_pages + (t / block_size)];
        uint token_in_block = t % block_size;
        device const float* head_k = k_cache + (block_idx * block_size * n_head_kv * head_dim) + (token_in_block * n_head_kv * head_dim) + (kv_head_idx * head_dim);
        
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) score += node_q[d] * head_k[d];
        score *= inv_sqrt_head_dim;

        float m_old = m;
        m = max(m_old, score);
        float exp_score = exp(score - m);
        float exp_m_diff = exp(m_old - m);
        l = l * exp_m_diff + exp_score;

        device const float* head_v = v_cache + (block_idx * block_size * n_head_kv * head_dim) + (token_in_block * n_head_kv * head_dim) + (kv_head_idx * head_dim);
        for (uint d = 0; d < head_dim; d++) acc[d] = acc[d] * exp_m_diff + exp_score * head_v[d];
    }

    // 2. Attend to Ancestors in the Tree
    // Note: This assumes the drafted tokens for the current tree are NOT yet in k_cache,
    // or they are passed in a separate buffer. 
    // To keep it a single pass, we'll assume they are in 'q' or a 'k_tree' buffer.
    // For now, let's assume 'verify_tree' will populate the KV cache for the tree nodes first.
    // Wait! If they are in KV cache, we need their positions.
    for (uint depth = 0; depth < max_tree_depth; depth++) {
        int ancestor_idx = ancestors_map[node_idx * max_tree_depth + depth];
        if (ancestor_idx == -1) break;
        if ((uint)ancestor_idx == node_idx) continue; // Don't attend to self? (Causal)

        // Find ancestor's token in KV cache
        uint t = n_ctx + depth; // Depth corresponds to position in the sequence
        uint block_idx = slot_mapping[slot_id * max_pages + (t / block_size)];
        uint token_in_block = t % block_size;
        
        device const float* head_k = k_cache + (block_idx * block_size * n_head_kv * head_dim) + (token_in_block * n_head_kv * head_dim) + (kv_head_idx * head_dim);
        // ... same softmax logic ...
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) score += node_q[d] * head_k[d];
        score *= inv_sqrt_head_dim;

        float m_old = m;
        m = max(m_old, score);
        float exp_score = exp(score - m);
        float exp_m_diff = exp(m_old - m);
        l = l * exp_m_diff + exp_score;

        device const float* head_v = v_cache + (block_idx * block_size * n_head_kv * head_dim) + (token_in_block * n_head_kv * head_dim) + (kv_head_idx * head_dim);
        for (uint d = 0; d < head_dim; d++) acc[d] = acc[d] * exp_m_diff + exp_score * head_v[d];
    }

    // 3. Finalize
    for (uint d = 0; d < head_dim; d++) node_out[d] = acc[d] / l;
}

/**
 * KV Cache Update (FP32).
 * 
 * Copies the Key and Value vectors for the current token into the global paged KV cache.
 * 
 * @param k_cache       Global Key cache buffer
 * @param v_cache       Global Value cache buffer
 * @param k_in          Input Key vector for current token
 * @param v_in          Input Value vector for current token
 * @param slot_id       ID of the active inference slot
 * @param slot_mapping  Buffer mapping slots to physical KV pages
 * @param max_pages     Total number of pages in the pool
 * @param block_size    Number of tokens per KV page
 * @param n_head        Number of KV heads
 * @param head_dim      Dimensionality of each KV head
 * @param pos           Current sequence position
 * @param tpig          Thread position in grid (maps to `head_idx * head_dim + dim_idx`)
 */
kernel void kv_update(
    device float* k_cache [[buffer(0)]],
    device float* v_cache [[buffer(1)]],
    device const float* k_in [[buffer(2)]],
    device const float* v_in [[buffer(3)]],
    constant uint& slot_id [[buffer(4)]],
    device const uint* slot_mapping [[buffer(5)]],
    constant uint& max_pages [[buffer(6)]],
    constant uint& block_size [[buffer(7)]],
    constant uint& n_head [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& pos [[buffer(10)]],
    uint tpig [[thread_position_in_grid]]
) {
    uint head_idx = tpig / head_dim;
    uint dim_idx = tpig % head_dim;
    if (head_idx >= n_head) return;

    uint block_idx = slot_mapping[slot_id * max_pages + (pos / block_size)];
    uint token_in_block = pos % block_size;

    uint cache_offset = (block_idx * block_size * n_head * head_dim) + (token_in_block * n_head * head_dim) + (head_idx * head_dim) + dim_idx;
    uint in_offset = (head_idx * head_dim) + dim_idx;

    k_cache[cache_offset] = k_in[in_offset];
    v_cache[cache_offset] = v_in[in_offset];
}

// ── Quantized Attention (INT8) ────────────────────────────────────────────────
// Cache layout: [Block][Token][Head][Dim] where Dim is char[head_dim] + float scale

/**
 * Flash Attention 2 for INT8 Paged KV Cache.
 * 
 * Computes attention scores where the KV cache is stored in 8-bit integers 
 * with a per-token/per-head FP32 scale factor. Unquantizes the keys and values 
 * on the fly during computation.
 * 
 * @param out           Output buffer for attention results [n_head, head_dim]
 * @param q             Query buffer for the current token [n_head, head_dim]
 * @param k_cache       Global INT8 Key cache buffer
 * @param v_cache       Global INT8 Value cache buffer
 * @param head_dim      Dimensionality of each attention head
 * @param n_ctx         Total context length (not used in tiling loop)
 * @param pos           Current sequence position
 * @param slot_id       ID of the active inference slot
 * @param slot_mapping  Buffer mapping slots to physical KV pages
 * @param max_pages     Total number of pages in the pool
 * @param block_size    Number of tokens per KV page
 * @param n_head_kv     Number of KV heads (for GQA)
 * @param n_head        Number of query heads
 * @param head_idx      Thread position mapping to the query head index
 */
kernel void paged_attention_flash_int8(
    device float* out [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const char* k_cache [[buffer(2)]],
    device const char* v_cache [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& n_ctx [[buffer(5)]],
    constant uint& pos [[buffer(6)]],
    constant uint& slot_id [[buffer(7)]],
    device const uint* slot_mapping [[buffer(8)]],
    constant uint& max_pages [[buffer(9)]],
    constant uint& block_size [[buffer(10)]],
    constant uint& n_head_kv [[buffer(11)]],
    constant uint& n_head [[buffer(12)]],
    uint head_idx [[thread_position_in_grid]]
) {
    uint kv_head_idx = head_idx / (n_head / n_head_kv);
    device const float* head_q = q + head_idx * head_dim;
    device float* head_out = out + head_idx * head_dim;

    float inv_sqrt_head_dim = 1.0f / sqrt((float)head_dim);
    uint head_bytes = head_dim + 4;
    
    float m = -INFINITY;
    float l = 0.0f;
    float acc[128];
    for (uint i = 0; i < head_dim; i++) acc[i] = 0.0f;

    uint total_tokens = pos + 1;

    for (uint t = 0; t < total_tokens; t++) {
        uint block_idx = slot_mapping[slot_id * max_pages + (t / block_size)];
        uint token_in_block = t % block_size;
        
        device const char* k_head_ptr = k_cache + (block_idx * block_size * n_head_kv * head_bytes) + (token_in_block * n_head_kv * head_bytes) + (kv_head_idx * head_bytes);
        float k_scale = *(device const float*)(k_head_ptr + head_dim);

        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += head_q[d] * ((float)k_head_ptr[d] * k_scale);
        }
        score *= inv_sqrt_head_dim;

        float m_old = m;
        m = max(m_old, score);
        float exp_score = exp(score - m);
        float exp_m_diff = exp(m_old - m);
        l = l * exp_m_diff + exp_score;

        device const char* v_head_ptr = v_cache + (block_idx * block_size * n_head_kv * head_bytes) + (token_in_block * n_head_kv * head_bytes) + (kv_head_idx * head_bytes);
        float v_scale = *(device const float*)(v_head_ptr + head_dim);
        
        for (uint d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * exp_m_diff + exp_score * ((float)v_head_ptr[d] * v_scale);
        }
    }

    for (uint d = 0; d < head_dim; d++) {
        head_out[d] = acc[d] / l;
    }
}

/**
 * KV Cache Update (INT8 Quantization).
 * 
 * Quantizes and copies the FP32 Key and Value vectors for the current token 
 * into the global INT8 paged KV cache. Calculates and stores a per-head scale factor.
 * 
 * @param k_cache       Global INT8 Key cache buffer
 * @param v_cache       Global INT8 Value cache buffer
 * @param k_in          Input FP32 Key vector for current token
 * @param v_in          Input FP32 Value vector for current token
 * @param slot_id       ID of the active inference slot
 * @param slot_mapping  Buffer mapping slots to physical KV pages
 * @param max_pages     Total number of pages in the pool
 * @param block_size    Number of tokens per KV page
 * @param n_head        Number of KV heads
 * @param head_dim      Dimensionality of each KV head
 * @param pos           Current sequence position
 * @param tpig          Thread position in grid (maps to `head_idx`)
 */
kernel void kv_update_int8(
    device char* k_cache [[buffer(0)]],
    device char* v_cache [[buffer(1)]],
    device const float* k_in [[buffer(2)]],
    device const float* v_in [[buffer(3)]],
    constant uint& slot_id [[buffer(4)]],
    device const uint* slot_mapping [[buffer(5)]],
    constant uint& max_pages [[buffer(6)]],
    constant uint& block_size [[buffer(7)]],
    constant uint& n_head [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& pos [[buffer(10)]],
    uint tpig [[thread_position_in_grid]] // thread per head
) {
    uint head_idx = tpig;
    if (head_idx >= n_head) return;

    uint block_idx = slot_mapping[slot_id * max_pages + (pos / block_size)];
    uint token_in_block = pos % block_size;

    uint head_bytes = head_dim + 4;
    uint cache_offset = (block_idx * block_size * n_head * head_bytes) + (token_in_block * n_head * head_bytes) + (head_idx * head_bytes);
    uint in_offset = head_idx * head_dim;

    device char* k_dst = k_cache + cache_offset;
    device char* v_dst = v_cache + cache_offset;
    device const float* k_src = k_in + in_offset;
    device const float* v_src = v_in + in_offset;

    // Find scales
    float k_max = 0.0f;
    float v_max = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        k_max = max(k_max, abs(k_src[d]));
        v_max = max(v_max, abs(v_src[d]));
    }

    float k_scale = k_max / 127.0f;
    float v_scale = v_max / 127.0f;
    float k_inv_scale = k_max > 0.0f ? 127.0f / k_max : 0.0f;
    float v_inv_scale = v_max > 0.0f ? 127.0f / v_max : 0.0f;

    for (uint d = 0; d < head_dim; d++) {
        k_dst[d] = (char)(k_src[d] * k_inv_scale);
        v_dst[d] = (char)(v_src[d] * v_inv_scale);
    }

    *(device float*)(k_dst + head_dim) = k_scale;
    *(device float*)(v_dst + head_dim) = v_scale;
}

// ── Element-wise Addition ─────────────────────────────────────────────────────

/**
 * Element-wise Vector Addition.
 * 
 * Computes `x[i] = x[i] + y[i]` for two vectors.
 * 
 * @param x     First vector buffer (modified in-place)
 * @param y     Second vector buffer (read-only)
 * @param tpig  Thread position in grid (maps to vector element index)
 */
kernel void vec_add(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    uint tpig [[thread_position_in_grid]]
) {
    x[tpig] += y[tpig];
}

// ── Sampling (ArgMax) ─────────────────────────────────────────────────────────

/**
 * Parallel ArgMax Reduction for GPU-Resident Sampling.
 * 
 * Finds the token index with the highest logit value using a threadgroup-level
 * reduction with SIMD shuffle primitives.
 * 
 * @param out_token      Result buffer for the winning TokenId
 * @param logits         Input logit buffer [n_vocab]
 * @param n              Vocab size (n_vocab)
 */
kernel void argmax(
    device uint* out_token [[buffer(0)]],
    device const float* logits [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tpig [[thread_position_in_grid]],
    uint tisl [[thread_index_in_simdgroup]],
    uint simds_per_tg [[simdgroups_per_threadgroup]],
    uint tpitg [[thread_position_in_threadgroup]],
    uint sidx [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float local_max[32]; 
    threadgroup uint local_idx[32];
    
    float val = (tpig < n) ? logits[tpig] : -INFINITY;
    uint idx = tpig;
    
    // SIMD-level reduction
    for (uint offset = 16; offset > 0; offset /= 2) {
        float other_val = simd_shuffle_down(val, offset);
        uint other_idx = simd_shuffle_down(idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    
    if (tisl == 0) {
        local_max[sidx] = val;
        local_idx[sidx] = idx;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tpitg == 0) {
        float final_max = -INFINITY;
        uint final_idx = 0;
        for (uint i = 0; i < simds_per_tg; i++) {
            if (local_max[i] > final_max) {
                final_max = local_max[i];
                final_idx = local_idx[i];
            }
        }
        *out_token = final_idx;
    }
}
