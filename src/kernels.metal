#include <metal_stdlib>
using namespace metal;

// ── RMS Norm ──────────────────────────────────────────────────────────────────

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

// ── Matrix-Vector Multiply (Q4_0) ─────────────────────────────────────────────
// Block: 2 bytes scale (f16) + 16 bytes nibbles (32 elements) = 18 bytes.
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

// ── Activation (SiLU) ─────────────────────────────────────────────────────────

kernel void silu(
    device float* x [[buffer(0)]],
    uint tpig [[thread_position_in_grid]]
) {
    float val = x[tpig];
    x[tpig] = val / (1.0f + exp(-val));
}

// ── Element-wise Multiplication ──────────────────────────────────────────────

kernel void vec_mul(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    uint tpig [[thread_position_in_grid]]
) {
    x[tpig] *= y[tpig];
}

// ── Softmax ───────────────────────────────────────────────────────────────────

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

kernel void attn_q_k(
    device float* scores [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k_cache [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& n_ctx [[buffer(4)]],
    constant uint& pos [[buffer(5)]],
    constant uint& slot_id [[buffer(6)]],
    device const uint* slot_mapping [[buffer(7)]],
    constant uint& max_pages [[buffer(8)]],
    constant uint& block_size [[buffer(9)]],
    constant uint& n_head [[buffer(10)]],
    uint tpig [[thread_position_in_grid]] // thread per head
) {
    uint head_idx = tpig;
    device const float* head_q = q + head_idx * head_dim;
    device float* head_scores = scores + head_idx * n_ctx;

    float inv_sqrt_head_dim = 1.0f / sqrt((float)head_dim);

    for (uint t = 0; t <= pos; t++) {
        float sum = 0.0f;
        uint block_idx = slot_mapping[slot_id * max_pages + (t / block_size)];
        uint token_in_block = t % block_size;
        device const float* head_k = k_cache + (block_idx * block_size * n_head * head_dim) + (token_in_block * n_head * head_dim) + (head_idx * head_dim);
        for (uint d = 0; d < head_dim; d++) {
            sum += head_q[d] * head_k[d];
        }
        head_scores[t] = sum * inv_sqrt_head_dim;
    }
}

kernel void attn_p_v(
    device float* out [[buffer(0)]],
    device const float* probs [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& n_ctx [[buffer(4)]],
    constant uint& pos [[buffer(5)]],
    constant uint& slot_id [[buffer(6)]],
    device const uint* slot_mapping [[buffer(7)]],
    constant uint& max_pages [[buffer(8)]],
    constant uint& block_size [[buffer(9)]],
    constant uint& n_head [[buffer(10)]],
    uint tpig [[thread_position_in_grid]] // thread per head
) {
    uint head_idx = tpig;
    device const float* head_probs = probs + head_idx * n_ctx;
    device float* head_out = out + head_idx * head_dim;

    for (uint d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (uint t = 0; t <= pos; t++) {
            uint block_idx = slot_mapping[slot_id * max_pages + (t / block_size)];
            uint token_in_block = t % block_size;
            device const float* head_v = v_cache + (block_idx * block_size * n_head * head_dim) + (token_in_block * n_head * head_dim) + (head_idx * head_dim);
            sum += head_probs[t] * head_v[d];
        }
        head_out[d] = sum;
    }
}

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

// ── Element-wise Addition ─────────────────────────────────────────────────────

kernel void vec_add(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    uint tpig [[thread_position_in_grid]]
) {
    x[tpig] += y[tpig];
}
