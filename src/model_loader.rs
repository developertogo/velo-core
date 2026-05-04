//! Model metadata loader.
//!
//! Opens a GGUF file, extracts `ModelMeta` (architecture dimensions), and
//! builds a `WeightStore` that maps tensor names to raw byte slices inside
//! a contiguous buffer.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::gguf::{GgmlType, GgufError, GgufFile, TensorInfo};
use crate::metal::Quantization;

// ── ModelMeta ─────────────────────────────────────────────────────────────────

/// Architecture dimensions extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelMeta {
    /// Canonical model architecture string (e.g. `"llama"`).
    pub arch: String,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// Maximum context length.
    pub n_ctx: usize,
    /// FFN intermediate (gate/up) dimension.
    pub n_ff: usize,
    pub head_dim: usize,
    pub rope_freq_base: f32,
    pub norm_eps: f32,
    pub quantization: Quantization,
}

impl ModelMeta {
    /// Parse model metadata from a `GgufFile`.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, LoadError> {
        let meta = &file.metadata;

        macro_rules! opt_f32 {
            ($key:expr, $default:expr) => {
                meta.get($key)
                    .and_then(|v| v.as_f32())
                    .unwrap_or($default)
            };
        }

        let arch = meta
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();

        let prefix = arch.as_str();

        let n_vocab = meta
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .or_else(|| {
                meta.get(&format!("{prefix}.vocab_size"))
                    .and_then(|v| v.as_u32())
                    .map(|v| v as usize)
            })
            .ok_or_else(|| LoadError::MissingMeta("vocab_size".to_string()))?;

        let n_embd = meta
            .get(&format!("{prefix}.embedding_length"))
            .and_then(|v| v.as_u32())
            .ok_or_else(|| LoadError::MissingMeta(format!("{prefix}.embedding_length")))? as usize;

        let n_layer = meta
            .get(&format!("{prefix}.block_count"))
            .and_then(|v| v.as_u32())
            .ok_or_else(|| LoadError::MissingMeta(format!("{prefix}.block_count")))? as usize;

        let n_head = meta
            .get(&format!("{prefix}.attention.head_count"))
            .and_then(|v| v.as_u32())
            .ok_or_else(|| LoadError::MissingMeta(format!("{prefix}.attention.head_count")))? as usize;

        let n_head_kv = meta
            .get(&format!("{prefix}.attention.head_count_kv"))
            .and_then(|v| v.as_u32())
            .unwrap_or(n_head as u32) as usize;

        let n_ctx = meta
            .get(&format!("{prefix}.context_length"))
            .and_then(|v| v.as_u32())
            .unwrap_or(4096) as usize;

        let n_ff = meta
            .get(&format!("{prefix}.feed_forward_length"))
            .and_then(|v| v.as_u32())
            .ok_or_else(|| LoadError::MissingMeta(format!("{prefix}.feed_forward_length")))? as usize;

        let head_dim = n_embd / n_head;

        let rope_freq_base =
            opt_f32!(&format!("{prefix}.rope.freq_base"), 10_000.0);

        let norm_eps =
            opt_f32!(&format!("{prefix}.attention.layer_norm_rms_epsilon"), 1e-5);

        // Infer quantization from the embedding weight type
        let quantization = infer_quantization(file);

        Ok(Self {
            arch,
            n_vocab,
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            n_ctx,
            n_ff,
            head_dim,
            rope_freq_base,
            norm_eps,
            quantization,
        })
    }
}

fn infer_quantization(file: &GgufFile) -> Quantization {
    let probe = file
        .tensors
        .get("blk.0.attn_q.weight")
        .or_else(|| file.tensors.get("token_embd.weight"))
        .map(|t| t.dtype);
    match probe {
        Some(GgmlType::Q4K) => Quantization::Q4K,
        _ => Quantization::Q4_0,
    }
}

// ── WeightStore ───────────────────────────────────────────────────────────────

/// Contiguous buffer holding all tensor data, with an index of named views.
pub struct WeightStore {
    pub meta: ModelMeta,
    /// Tensor metadata indexed by name.
    pub index: HashMap<String, TensorInfo>,
    /// Raw tensor bytes (all tensors concatenated, data_offset-based).
    pub data: Vec<u8>,
}

impl WeightStore {
    /// Byte slice for `name`, or `None` if absent.
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        let info = self.index.get(name)?;
        let start = info.offset as usize;
        let end = start + info.nbytes as usize;
        self.data.get(start..end)
    }

    /// Byte slice for `name`, returning an error if missing.
    pub fn require(&self, name: &str) -> Result<&[u8], LoadError> {
        self.get(name)
            .ok_or_else(|| LoadError::MissingTensor(name.to_string()))
    }

    /// Dtype of named tensor.
    pub fn dtype(&self, name: &str) -> Option<GgmlType> {
        self.index.get(name).map(|t| t.dtype)
    }

    /// Shape of named tensor.
    pub fn shape(&self, name: &str) -> Option<&[u64]> {
        self.index.get(name).map(|t| t.shape.as_slice())
    }

    /// Validate that all expected tensors for the architecture are present and have correct shapes.
    pub fn validate(&self) -> Result<(), LoadError> {
        let meta = &self.meta;
        let n_embd = meta.n_embd as u64;
        let n_vocab = meta.n_vocab as u64;
        let n_layer = meta.n_layer;
        let n_ff = meta.n_ff as u64;
        let n_head_kv = meta.n_head_kv as u64;
        let head_dim = meta.head_dim as u64;

        let check = |name: &str, expected: &[u64]| {
            let info = self.index.get(name).ok_or_else(|| LoadError::MissingTensor(name.to_string()))?;
            // GGUF shapes are usually [cols, rows] or [cols, rows, layers]
            // We expect the last dimensions to match our logic.
            // For matvec, we usually have [n_embd, rows]
            if info.shape != expected {
                return Err(LoadError::InvalidShape {
                    tensor: name.to_string(),
                    got: info.shape.clone(),
                    expected: format!("{:?}", expected).leak(), // Simplified leak for static str requirement
                });
            }
            Ok(())
        };

        check("token_embd.weight", &[n_embd, n_vocab])?;
        
        for l in 0..n_layer {
            check(&format!("layers.{}.attention_norm.weight", l), &[n_embd])?;
            check(&format!("layers.{}.attention.wq.weight", l), &[n_embd, n_embd])?;
            check(&format!("layers.{}.attention.wk.weight", l), &[n_embd, n_head_kv * head_dim])?;
            check(&format!("layers.{}.attention.wv.weight", l), &[n_embd, n_head_kv * head_dim])?;
            check(&format!("layers.{}.attention.wo.weight", l), &[n_embd, n_embd])?;
            
            check(&format!("layers.{}.ffn_norm.weight", l), &[n_embd])?;
            check(&format!("layers.{}.feed_forward.w1.weight", l), &[n_embd, n_ff])?;
            check(&format!("layers.{}.feed_forward.w2.weight", l), &[n_ff, n_embd])?;
            check(&format!("layers.{}.feed_forward.w3.weight", l), &[n_embd, n_ff])?;
        }

        check("output_norm.weight", &[n_embd])?;
        check("output.weight", &[n_embd, n_vocab])?;

        Ok(())
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Load a GGUF model from disk.
///
/// Reads all tensor data into memory. For 8 B-class models this is ~4–5 GB
/// (Q4) or ~16 GB (F16); ensure sufficient RAM before calling.
pub fn load_gguf(path: &Path) -> Result<WeightStore, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::Io(e, path.display().to_string()))?;
    let mut reader = BufReader::new(file);

    let gguf = GgufFile::read(&mut reader).map_err(LoadError::Gguf)?;
    let meta = ModelMeta::from_gguf(&gguf)?;

    let data = gguf
        .read_all_tensors(&mut reader)
        .map_err(LoadError::Gguf)?;

    let store = WeightStore {
        meta,
        index: gguf.tensors,
        data,
    };
    store.validate()?;
    Ok(store)
}

// ── LoadError ─────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error, String),
    Gguf(GgufError),
    MissingMeta(String),
    MissingTensor(String),
    InvalidShape { tensor: String, got: Vec<u64>, expected: &'static str },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e, p) => write!(f, "I/O error loading {p}: {e}"),
            Self::Gguf(e) => write!(f, "GGUF parse error: {e}"),
            Self::MissingMeta(k) => write!(f, "missing metadata key: {k}"),
            Self::MissingTensor(k) => write!(f, "missing tensor: {k}"),
            Self::InvalidShape { tensor, got, expected } => {
                write!(f, "tensor {tensor}: expected shape {expected}, got {got:?}")
            }
        }
    }
}

impl std::error::Error for LoadError {}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufValue;
    use std::collections::HashMap;

    fn llama_meta_map() -> HashMap<String, GgufValue> {
        let mut m = HashMap::new();
        m.insert("general.architecture".into(), GgufValue::String("llama".into()));
        m.insert("llama.embedding_length".into(), GgufValue::U32(4096));
        m.insert("llama.block_count".into(), GgufValue::U32(32));
        m.insert("llama.attention.head_count".into(), GgufValue::U32(32));
        m.insert("llama.attention.head_count_kv".into(), GgufValue::U32(8));
        m.insert("llama.context_length".into(), GgufValue::U32(8192));
        m.insert("llama.feed_forward_length".into(), GgufValue::U32(14336));
        m.insert("llama.rope.freq_base".into(), GgufValue::F32(500_000.0));
        // vocab via token array
        let tokens: Vec<GgufValue> = (0..128256)
            .map(|_| GgufValue::String(String::new()))
            .collect();
        m.insert("tokenizer.ggml.tokens".into(), GgufValue::Array(tokens));
        m
    }

    fn fake_gguf(meta: HashMap<String, GgufValue>) -> GgufFile {
        GgufFile {
            version: 3,
            metadata: meta,
            tensors: HashMap::new(),
            data_offset: 0,
        }
    }

    #[test]
    fn parses_llama3_8b_dimensions() {
        let gguf = fake_gguf(llama_meta_map());
        let meta = ModelMeta::from_gguf(&gguf).unwrap();
        assert_eq!(meta.arch, "llama");
        assert_eq!(meta.n_vocab, 128256);
        assert_eq!(meta.n_embd, 4096);
        assert_eq!(meta.n_layer, 32);
        assert_eq!(meta.n_head, 32);
        assert_eq!(meta.n_head_kv, 8);
        assert_eq!(meta.n_ff, 14336);
        assert_eq!(meta.head_dim, 128);
        assert!((meta.rope_freq_base - 500_000.0).abs() < 1.0);
    }

    #[test]
    fn missing_embedding_length_is_error() {
        let mut m = llama_meta_map();
        m.remove("llama.embedding_length");
        let gguf = fake_gguf(m);
        assert!(ModelMeta::from_gguf(&gguf).is_err());
    }

    #[test]
    fn missing_block_count_is_error() {
        let mut m = llama_meta_map();
        m.remove("llama.block_count");
        let gguf = fake_gguf(m);
        assert!(ModelMeta::from_gguf(&gguf).is_err());
    }

    #[test]
    fn defaults_n_head_kv_to_n_head_when_absent() {
        let mut m = llama_meta_map();
        m.remove("llama.attention.head_count_kv");
        let gguf = fake_gguf(m);
        let meta = ModelMeta::from_gguf(&gguf).unwrap();
        assert_eq!(meta.n_head_kv, meta.n_head);
    }

    #[test]
    fn validation_fails_on_missing_tensors() {
        let meta = ModelMeta::from_gguf(&fake_gguf(llama_meta_map())).unwrap();
        let store = WeightStore {
            meta,
            index: HashMap::new(),
            data: Vec::new(),
        };
        assert!(store.validate().is_err());
    }

    #[test]
    fn validation_fails_on_wrong_shape() {
        let mut m = llama_meta_map();
        m.insert("llama.block_count".into(), GgufValue::U32(1)); // 1 layer only
        let meta = ModelMeta::from_gguf(&fake_gguf(m)).unwrap();
        
        let mut tensors = HashMap::new();
        // Missing almost all tensors, but let's just add one with wrong shape
        tensors.insert("token_embd.weight".to_string(), TensorInfo {
            name: "token_embd.weight".to_string(),
            dtype: GgmlType::F32,
            shape: vec![1, 1], // Wrong
            offset: 0,
            nbytes: 4,
        });

        let store = WeightStore {
            meta,
            index: tensors,
            data: vec![0, 0, 0, 0],
        };
        
        let err = store.validate().unwrap_err();
        assert!(err.to_string().contains("expected shape"));
    }
}
