//! LLaMA CPU inference engine.
//!
//! Implements a complete transformer forward pass on CPU using dequantized
//! weights. Supports LLaMA 2 / LLaMA 3 / Mistral architectures (GQA, SwiGLU,
//! RoPE).
//!
//! # Design
//!
//! * Weights are dequantized lazily per layer to bound peak memory.
//! * A flat `Vec<f32>` KV cache is maintained per layer.
//! * The engine is single-threaded; parallelism is left to the caller.

use crate::backend::{CausalLmBackend, TokenLogits};
use crate::model_loader::{LoadError, ModelMeta, WeightStore};
use crate::quant::dequant_row;
use crate::radix_cache::{CacheLookup, TokenId};
use crate::speculative::{Result as SpecResult, SpeculativeError};
use crate::gguf::GgmlType;

// ── KV cache ──────────────────────────────────────────────────────────────────

struct KvCache {
    n_layer: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_ctx: usize,
    /// k[layer][pos * n_head_kv * head_dim + head * head_dim + d]
    k: Vec<Vec<f32>>,
    /// v[layer][pos * n_head_kv * head_dim + head * head_dim + d]
    v: Vec<Vec<f32>>,
    pos: usize,
}

impl KvCache {
    fn new(n_layer: usize, n_head_kv: usize, head_dim: usize, n_ctx: usize) -> Self {
        let per_layer = n_ctx * n_head_kv * head_dim;
        Self {
            n_layer,
            n_head_kv,
            head_dim,
            n_ctx,
            k: vec![vec![0.0; per_layer]; n_layer],
            v: vec![vec![0.0; per_layer]; n_layer],
            pos: 0,
        }
    }

    fn store_kv(&mut self, layer: usize, pos: usize, k: &[f32], v_vec: &[f32]) {
        let stride = self.n_head_kv * self.head_dim;
        let start = pos * stride;
        self.k[layer][start..start + stride].copy_from_slice(k);
        self.v[layer][start..start + stride].copy_from_slice(v_vec);
    }
}

// ── Math primitives ───────────────────────────────────────────────────────────

/// RMS normalisation: `x = x / rms(x) * w`
fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for (xi, wi) in x.iter_mut().zip(w.iter()) {
        *xi = (*xi / rms) * wi;
    }
}

/// Matrix-vector multiply: `out[r] += sum_c W[r,c] * x[c]` (no bias).
///
/// `w` is a **dequantised** row-major matrix of shape `[rows, cols]`.
fn matvec(out: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        out[r] += row.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f32>();
    }
}

/// Quantised matrix-vector multiply — dequantises one row at a time.
fn matvec_quant(
    out: &mut [f32],
    data: &[u8],
    dtype: GgmlType,
    x: &[f32],
    rows: usize,
    cols: usize,
) {
    let row_bytes = dtype.nbytes(cols as u64) as usize;
    let mut row_f32 = vec![0.0f32; cols];
    for r in 0..rows {
        let src = &data[r * row_bytes..(r + 1) * row_bytes];
        dequant_row(dtype, src, &mut row_f32);
        out[r] += row_f32.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f32>();
    }
}

/// SiLU activation: `x * sigmoid(x)`.
#[inline]
fn silu(x: f32) -> f32 { x * (1.0 / (1.0 + (-x).exp())) }

/// Rotary position embedding applied in-place to `q` and `k`.
///
/// Both are shaped as `[n_heads, head_dim]` (or `n_kv_heads` for `k`).
fn rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    freq_base: f32,
) {
    let half = head_dim / 2;
    apply_rope(q, pos, n_heads, head_dim, half, freq_base);
    apply_rope(k, pos, n_kv_heads, head_dim, half, freq_base);
}

fn apply_rope(
    x: &mut [f32],
    pos: usize,
    n_heads: usize,
    head_dim: usize,
    half: usize,
    freq_base: f32,
) {
    for h in 0..n_heads {
        let offset = h * head_dim;
        for i in 0..half {
            let theta = pos as f32 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
            let (sin_t, cos_t) = theta.sin_cos();
            let x0 = x[offset + i];
            let x1 = x[offset + i + half];
            x[offset + i] = x0 * cos_t - x1 * sin_t;
            x[offset + i + half] = x0 * sin_t + x1 * cos_t;
        }
    }
}

/// Softmax in-place over `x`.
fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

// ── Weight lookup helpers ─────────────────────────────────────────────────────

/// A borrowed tensor with its dtype, used for quantised mat-vec.
struct TensorRef<'a> {
    data: &'a [u8],
    dtype: GgmlType,
}

/// Dequantise a 1-D weight tensor (e.g. norm weights) into `out`.
fn load_vec(weights: &WeightStore, name: &str, out: &mut Vec<f32>) -> Result<(), LoadError> {
    let data = weights.require(name)?;
    let dtype = weights.dtype(name).unwrap();
    let n = out.len();
    dequant_row(dtype, data, out);
    let _ = n;
    Ok(())
}

// ── LlamaCpuModel ─────────────────────────────────────────────────────────────

/// CPU-backed LLaMA inference model.
///
/// Holds a reference to the `WeightStore` and a KV cache. Implements
/// `CausalLmBackend` so it can be used directly by `GreedyDraftModel` /
/// `GreedyTargetModel`.
pub struct LlamaCpuModel {
    meta: ModelMeta,
    weights: WeightStore,
    kv: KvCache,
    bound_prefix: Option<CacheLookup>,
}

impl LlamaCpuModel {
    /// Construct from a loaded `WeightStore`.
    pub fn new(weights: WeightStore) -> Self {
        let meta = weights.meta.clone();
        let kv = KvCache::new(
            meta.n_layer,
            meta.n_head_kv,
            meta.head_dim,
            meta.n_ctx,
        );
        Self { meta, weights, kv, bound_prefix: None }
    }

    /// Forward pass for a single token at position `pos`.
    ///
    /// Returns logits over the full vocabulary.
    pub fn forward_one(
        &mut self,
        token: TokenId,
        pos: usize,
    ) -> Result<Vec<f32>, LoadError> {
        let meta = &self.meta;
        let n_embd = meta.n_embd;
        let n_head = meta.n_head;
        let n_head_kv = meta.n_head_kv;
        let head_dim = meta.head_dim;
        let n_ff = meta.n_ff;
        let n_vocab = meta.n_vocab;
        let freq_base = meta.rope_freq_base;
        let norm_eps = meta.norm_eps;

        // ── Token embedding lookup ────────────────────────────────────────────
        let embd_data = self.weights.require("token_embd.weight")?;
        let embd_dtype = self.weights.dtype("token_embd.weight").unwrap();
        let embd_row_bytes = embd_dtype.nbytes(n_embd as u64) as usize;
        let embd_start = token as usize * embd_row_bytes;
        let mut x = vec![0.0f32; n_embd];
        dequant_row(embd_dtype, &embd_data[embd_start..embd_start + embd_row_bytes], &mut x);

        // ── Transformer layers ────────────────────────────────────────────────
        for layer in 0..meta.n_layer {
            let l = layer;

            // Attention norm
            let atn_norm_name = format!("blk.{l}.attn_norm.weight");
            let mut norm_w = vec![0.0f32; n_embd];
            load_vec(&self.weights, &atn_norm_name, &mut norm_w)?;

            let mut xb = x.clone();
            rms_norm(&mut xb, &norm_w, norm_eps);

            // Q, K, V projections
            let q_name = format!("blk.{l}.attn_q.weight");
            let k_name = format!("blk.{l}.attn_k.weight");
            let v_name = format!("blk.{l}.attn_v.weight");

            let mut q = vec![0.0f32; n_head * head_dim];
            let mut k_vec = vec![0.0f32; n_head_kv * head_dim];
            let mut v_vec = vec![0.0f32; n_head_kv * head_dim];

            {
                let (data, dtype) = self.weights_ref(&q_name)?;
                matvec_quant(&mut q, data, dtype, &xb, n_head * head_dim, n_embd);
            }
            {
                let (data, dtype) = self.weights_ref(&k_name)?;
                matvec_quant(&mut k_vec, data, dtype, &xb, n_head_kv * head_dim, n_embd);
            }
            {
                let (data, dtype) = self.weights_ref(&v_name)?;
                matvec_quant(&mut v_vec, data, dtype, &xb, n_head_kv * head_dim, n_embd);
            }

            // RoPE
            rope(&mut q, &mut k_vec, pos, n_head, n_head_kv, head_dim, freq_base);

            // Store KV
            self.kv.store_kv(layer, pos, &k_vec, &v_vec);

            // Grouped-query attention
            let mut x_attn = vec![0.0f32; n_head * head_dim];
            let kv_stride = n_head_kv * head_dim;
            let n_seq = pos + 1;

            for h in 0..n_head {
                let kv_head = h * n_head_kv / n_head; // GQA mapping
                let q_h = &q[h * head_dim..(h + 1) * head_dim];

                // Attention scores
                let mut scores = vec![0.0f32; n_seq];
                for t in 0..n_seq {
                    let k_t = &self.kv.k[layer]
                        [t * kv_stride + kv_head * head_dim..t * kv_stride + kv_head * head_dim + head_dim];
                    let dot: f32 = q_h.iter().zip(k_t.iter()).map(|(a, b)| a * b).sum();
                    scores[t] = dot / (head_dim as f32).sqrt();
                }

                softmax(&mut scores);

                // Weighted sum of values
                let out_h = &mut x_attn[h * head_dim..(h + 1) * head_dim];
                for t in 0..n_seq {
                    let v_t = &self.kv.v[layer]
                        [t * kv_stride + kv_head * head_dim..t * kv_stride + kv_head * head_dim + head_dim];
                    let s = scores[t];
                    for (o, v) in out_h.iter_mut().zip(v_t.iter()) {
                        *o += s * v;
                    }
                }
            }

            // Output projection
            let o_name = format!("blk.{l}.attn_output.weight");
            let mut x_out = vec![0.0f32; n_embd];
            {
                let (data, dtype) = self.weights_ref(&o_name)?;
                matvec_quant(&mut x_out, data, dtype, &x_attn, n_embd, n_head * head_dim);
            }

            // Residual
            for (xi, oi) in x.iter_mut().zip(x_out.iter()) {
                *xi += oi;
            }

            // ── FFN ───────────────────────────────────────────────────────────
            let ffn_norm_name = format!("blk.{l}.ffn_norm.weight");
            let mut ffn_norm_w = vec![0.0f32; n_embd];
            load_vec(&self.weights, &ffn_norm_name, &mut ffn_norm_w)?;

            let mut xb2 = x.clone();
            rms_norm(&mut xb2, &ffn_norm_w, norm_eps);

            let gate_name = format!("blk.{l}.ffn_gate.weight");
            let up_name = format!("blk.{l}.ffn_up.weight");
            let down_name = format!("blk.{l}.ffn_down.weight");

            let mut gate = vec![0.0f32; n_ff];
            let mut up = vec![0.0f32; n_ff];
            {
                let (data, dtype) = self.weights_ref(&gate_name)?;
                matvec_quant(&mut gate, data, dtype, &xb2, n_ff, n_embd);
            }
            {
                let (data, dtype) = self.weights_ref(&up_name)?;
                matvec_quant(&mut up, data, dtype, &xb2, n_ff, n_embd);
            }

            // SwiGLU: gate = silu(gate) * up
            for (g, u) in gate.iter_mut().zip(up.iter()) {
                *g = silu(*g) * u;
            }

            let mut ffn_out = vec![0.0f32; n_embd];
            {
                let (data, dtype) = self.weights_ref(&down_name)?;
                matvec_quant(&mut ffn_out, data, dtype, &gate, n_embd, n_ff);
            }

            for (xi, fi) in x.iter_mut().zip(ffn_out.iter()) {
                *xi += fi;
            }
        }

        // ── Output head ───────────────────────────────────────────────────────
        let out_norm_data = self.weights.require("output_norm.weight")?;
        let out_norm_dtype = self.weights.dtype("output_norm.weight").unwrap();
        let mut out_norm_w = vec![0.0f32; n_embd];
        dequant_row(out_norm_dtype, out_norm_data, &mut out_norm_w);
        rms_norm(&mut x, &out_norm_w, norm_eps);

        // LM head (may be tied to token_embd)
        let lm_head_name = if self.weights.index.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let mut logits = vec![0.0f32; n_vocab];
        {
            let (data, dtype) = self.weights_ref(lm_head_name)?;
            matvec_quant(&mut logits, data, dtype, &x, n_vocab, n_embd);
        }

        Ok(logits)
    }

    /// Forward pass over a sequence of tokens (e.g. a full prompt).
    ///
    /// Returns logits for the **last** token position.
    pub fn forward_sequence(&mut self, tokens: &[TokenId]) -> Result<Vec<f32>, LoadError> {
        let start_pos = self.kv.pos;
        let mut last_logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            last_logits = self.forward_one(tok, pos)?;
        }
        self.kv.pos += tokens.len();
        Ok(last_logits)
    }

    fn weights_ref(&self, name: &str) -> Result<(&[u8], GgmlType), LoadError> {
        let data = self.weights.require(name)?;
        let dtype = self.weights.dtype(name).unwrap();
        Ok((data, dtype))
    }
}

// ── CausalLmBackend impl ──────────────────────────────────────────────────────

impl CausalLmBackend for LlamaCpuModel {
    fn bind_prefix_cache(&mut self, prefix: &CacheLookup) -> SpecResult<()> {
        self.bound_prefix = Some(prefix.clone());
        Ok(())
    }

    fn next_logits(&mut self, context: &[TokenId]) -> SpecResult<TokenLogits> {
        let last = context.last().copied().ok_or_else(|| {
            SpeculativeError::Model("next_logits called with empty context".to_string())
        })?;
        let pos = self.kv.pos;
        let logits = self
            .forward_one(last, pos)
            .map_err(|e| SpeculativeError::Model(e.to_string()))?;
        self.kv.pos += 1;
        TokenLogits::new(logits)
    }

    fn verify_logits(
        &mut self,
        context: &[TokenId],
        drafted: &[TokenId],
    ) -> SpecResult<Vec<TokenLogits>> {
        // Full batch verification: feed context + each drafted token
        let mut result = Vec::with_capacity(drafted.len());
        let base_pos = self.kv.pos;
        for (i, &tok) in drafted.iter().enumerate() {
            let pos = base_pos + i;
            let logits = self
                .forward_one(tok, pos)
                .map_err(|e| SpeculativeError::Model(e.to_string()))?;
            result.push(TokenLogits::new(logits)?);
        }
        let _ = context;
        self.kv.pos += drafted.len();
        Ok(result)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_weights_normalises() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];
        rms_norm(&mut x, &w, 1e-6);
        let rms: f32 = (x.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        // After normalisation the RMS should be ≈ 1.0
        assert!((rms - 1.0).abs() < 1e-4, "rms = {rms}");
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_max_element_is_largest() {
        let mut x = vec![0.1f32, 10.0, 0.1];
        softmax(&mut x);
        assert!(x[1] > x[0] && x[1] > x[2]);
    }

    #[test]
    fn silu_at_zero_is_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn silu_positive_is_positive() {
        assert!(silu(1.0) > 0.0);
    }

    #[test]
    fn matvec_identity() {
        // W = I (2×2), x = [3, 4] → out = [3, 4]
        let w = vec![1.0f32, 0.0, 0.0, 1.0];
        let x = vec![3.0f32, 4.0];
        let mut out = vec![0.0f32; 2];
        matvec(&mut out, &w, &x, 2, 2);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn rope_does_not_change_norm() {
        let mut q = vec![1.0f32, 0.0, 0.0, 1.0]; // 1 head, head_dim=4
        let mut k = vec![0.5f32, 0.5, -0.5, -0.5];
        let norm_q_before: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        rope(&mut q, &mut k, 0, 1, 1, 4, 10_000.0);
        let norm_q_after: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_q_before - norm_q_after).abs() < 1e-4);
    }
}
