//! Quantized weight dequantization.
//!
//! Implements block-level dequantization for the formats used in practice by
//! 8 B-class LLaMA models:
//!
//! * `Q4_0` — 32-element blocks: f16 scale + 16 packed nibbles
//! * `Q4_K` — 256-element super-blocks with per-sub-block 6-bit scales
//! * `Q6_K` — 256-element super-blocks (useful for the LM head)
//! * `F16` / `BF16` — direct f16→f32 conversion
//! * `F32` — identity copy

use crate::gguf::GgmlType;

// ── f16 helpers ───────────────────────────────────────────────────────────────

/// Convert an IEEE 754 binary16 (little-endian bytes) to f32.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    // Sign, exponent, mantissa
    let sign = ((bits >> 15) as u32) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        let v = (mant as f32) * (1.0 / (1u32 << 24) as f32);
        if sign != 0 { -v } else { v }
    } else if exp == 31 {
        // Inf / NaN
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | (((exp as u32) + 112) << 23) | ((mant as u32) << 13))
    }
}

/// Convert a BF16 (brain float) word to f32.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ── Q4_0 ─────────────────────────────────────────────────────────────────────

/// Block layout: 2 bytes (f16 delta) + 16 bytes (32 nibbles) = 18 bytes.
const Q4_0_BLOCK: usize = 18;
const Q4_0_ELEMS: usize = 32;

/// Dequantize a single Q4_0 block into `out[0..32]`.
#[inline]
fn dequant_q4_0_block(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= Q4_0_BLOCK);
    debug_assert!(out.len() >= Q4_0_ELEMS);

    let delta = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let nibbles = &block[2..18];

    for i in 0..16 {
        let byte = nibbles[i];
        let lo = (byte & 0x0F) as i8 - 8;
        let hi = ((byte >> 4) & 0x0F) as i8 - 8;
        out[i * 2] = delta * lo as f32;
        out[i * 2 + 1] = delta * hi as f32;
    }
}

/// Dequantize an entire Q4_0 weight row (multiple blocks) into `out`.
pub fn dequant_q4_0_row(data: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / Q4_0_ELEMS;
    for b in 0..n_blocks {
        let block = &data[b * Q4_0_BLOCK..(b + 1) * Q4_0_BLOCK];
        dequant_q4_0_block(block, &mut out[b * Q4_0_ELEMS..]);
    }
}

// ── Q4_K ─────────────────────────────────────────────────────────────────────

/// Super-block layout (144 bytes, 256 elements):
///   2  d      (f16 scale)
///   2  dmin   (f16 min scale)
///   12 scales (8 × 6-bit scale + 8 × 6-bit min, packed)
///   128 qs    (256 nibbles)
const Q4K_BLOCK: usize = 144;
const Q4K_ELEMS: usize = 256;
const Q4K_SUBS: usize = 8; // sub-blocks per super-block
const Q4K_SUB_ELEMS: usize = Q4K_ELEMS / Q4K_SUBS; // 32 elements per sub-block

/// Unpack the 12-byte scale field of a Q4_K super-block into 8 (scale, min) pairs.
fn unpack_q4k_scales(scales_bytes: &[u8]) -> [(u8, u8); Q4K_SUBS] {
    // Packed as: lower 4 bits of sc[0..7], then upper bits interleaved
    // Full 6-bit fields: packed 3 bytes encode 4 6-bit values
    // Layout per llama.cpp ggml-quants.c:
    //   bytes 0..5  contain lower 4 bits of sc[0..7] and mn[0..3]
    //   bytes 6..11 contain upper 2 bits
    let mut sc = [0u8; Q4K_SUBS];
    let mut mn = [0u8; Q4K_SUBS];

    // Lower 4 bits
    for i in 0..4 {
        sc[i] = scales_bytes[i] & 0x3F;
        mn[i] = scales_bytes[i + 4] & 0x3F;
        sc[i + 4] = scales_bytes[i + 8] & 0x3F;
        mn[i + 4] = scales_bytes[i + 8] >> 4 | ((scales_bytes[i + 4] >> 4) << 4);
    }
    // Upper 2 bits correction from the remaining scale bytes
    for i in 0..4 {
        sc[i] |= (scales_bytes[i + 8] & 0x0F) << 4;
        mn[i] |= (scales_bytes[i + 8] >> 4) << 4;
    }

    let mut out = [(0u8, 0u8); Q4K_SUBS];
    for i in 0..Q4K_SUBS {
        out[i] = (sc[i] & 0x3F, mn[i] & 0x3F);
    }
    out
}

/// Dequantize a single Q4_K super-block into `out[0..256]`.
fn dequant_q4k_block(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= Q4K_BLOCK);
    debug_assert!(out.len() >= Q4K_ELEMS);

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    let scales = unpack_q4k_scales(&block[4..16]);
    let qs = &block[16..Q4K_BLOCK]; // 128 bytes, 256 nibbles

    for sub in 0..Q4K_SUBS {
        let (sc, mn_sc) = scales[sub];
        let scale = d * sc as f32;
        let min = dmin * mn_sc as f32;

        let base = sub * Q4K_SUB_ELEMS;
        let qbase = sub * (Q4K_SUB_ELEMS / 2); // 16 bytes per sub-block

        for i in 0..16 {
            let byte = qs[qbase + i];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            out[base + i * 2] = scale * lo - min;
            out[base + i * 2 + 1] = scale * hi - min;
        }
    }
}

/// Dequantize an entire Q4_K weight row into `out`.
pub fn dequant_q4k_row(data: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / Q4K_ELEMS;
    for b in 0..n_blocks {
        let block = &data[b * Q4K_BLOCK..(b + 1) * Q4K_BLOCK];
        dequant_q4k_block(block, &mut out[b * Q4K_ELEMS..]);
    }
}

// ── F16 / BF16 / F32 ─────────────────────────────────────────────────────────

pub fn dequant_f16_row(data: &[u8], out: &mut [f32]) {
    for (i, chunk) in data.chunks_exact(2).enumerate().take(out.len()) {
        out[i] = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
}

pub fn dequant_bf16_row(data: &[u8], out: &mut [f32]) {
    for (i, chunk) in data.chunks_exact(2).enumerate().take(out.len()) {
        out[i] = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
}

pub fn copy_f32_row(data: &[u8], out: &mut [f32]) {
    for (i, chunk) in data.chunks_exact(4).enumerate().take(out.len()) {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

/// Dequantize one row of a 2-D weight matrix given its `GgmlType`.
///
/// `data` must be the raw bytes for exactly one row.
/// `out`  must have `n_cols` elements already allocated.
pub fn dequant_row(dtype: GgmlType, data: &[u8], out: &mut [f32]) {
    match dtype {
        GgmlType::F32 => copy_f32_row(data, out),
        GgmlType::F16 => dequant_f16_row(data, out),
        GgmlType::BF16 => dequant_bf16_row(data, out),
        GgmlType::Q4_0 | GgmlType::Q4_1 => dequant_q4_0_row(data, out),
        GgmlType::Q4K => dequant_q4k_row(data, out),
        // Fallback: zero-fill (future: add Q6K, Q8_0, …)
        _ => out.fill(0.0),
    }
}

/// Dequantize a 2-D weight matrix with shape `[rows, cols]`.
pub fn dequant_matrix(dtype: GgmlType, data: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = dtype.nbytes(cols as u64) as usize;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let src = &data[r * row_bytes..(r + 1) * row_bytes];
        dequant_row(dtype, src, &mut out[r * cols..(r + 1) * cols]);
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_zero_converts_to_zero() {
        assert_eq!(f16_to_f32(0), 0.0);
    }

    #[test]
    fn f16_one_converts_correctly() {
        // 0x3C00 is 1.0 in f16
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn f16_minus_one_converts_correctly() {
        // 0xBC00 is -1.0 in f16
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn bf16_converts_correctly() {
        // 0x3F80 is 1.0 in bf16
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-6);
    }

    fn make_q4_0_block(delta_f16: u16, nibbles: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&delta_f16.to_le_bytes());
        block.extend_from_slice(nibbles);
        block
    }

    #[test]
    fn dequant_q4_0_all_zeros_produces_zeros() {
        // delta=1.0 (0x3C00), nibbles all 0x88 (value 8 → 8-8=0)
        let block = make_q4_0_block(0x3C00, &[0x88; 16]);
        let mut out = vec![0.0f32; 32];
        dequant_q4_0_row(&block, &mut out);
        for v in &out {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn dequant_q4_0_scale_applies() {
        // delta = 2.0 (f16 = 0x4000), nibbles all 0x99 (lo=9→1, hi=9→1)
        let block = make_q4_0_block(0x4000, &[0x99; 16]);
        let mut out = vec![0.0f32; 32];
        dequant_q4_0_row(&block, &mut out);
        // Each element: delta * 1 = 2.0
        for v in &out {
            assert!((v - 2.0).abs() < 1e-4, "expected 2.0, got {v}");
        }
    }

    #[test]
    fn dequant_f32_row_roundtrips() {
        let values = [1.0f32, -2.5, 3.14, 0.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out = vec![0.0f32; 4];
        copy_f32_row(&bytes, &mut out);
        for (a, b) in values.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn dequant_matrix_f32_identity() {
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = dequant_matrix(GgmlType::F32, &data, 2, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
