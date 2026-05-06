use velo_core::paged_attention::KvCacheType;
use velo_core::model_loader::ModelMeta;
use velo_core::metal::MetalBackendConfig;

#[test]
fn test_kv_bytes_calculation() {
    let meta = ModelMeta {
        arch: "llama".to_string(),
        n_vocab: 32000,
        n_embd: 4096,
        n_layer: 32,
        n_head: 32,
        n_head_kv: 32,
        n_ctx: 4096,
        n_ff: 11008,
        head_dim: 128,
        rope_freq_base: 10000.0,
        norm_eps: 1e-5,
        quantization: velo_core::metal::Quantization::F32,
    };

    // FP32: 128 * 4 = 512 bytes per head
    assert_eq!(meta.kv_bytes_per_token_per_head(KvCacheType::Fp32), 512);
    // INT8: 128 + 4 = 132 bytes per head
    assert_eq!(meta.kv_bytes_per_token_per_head(KvCacheType::Int8), 132);

    // Total for 32 heads
    assert_eq!(meta.kv_bytes_per_token(KvCacheType::Fp32), 32 * 512);
    assert_eq!(meta.kv_bytes_per_token(KvCacheType::Int8), 32 * 132);
}

#[test]
fn test_metal_backend_compression_config() {
    let mut config = MetalBackendConfig::default();
    config.kv_type = KvCacheType::Int8;
    assert_eq!(config.kv_type, KvCacheType::Int8);
}
