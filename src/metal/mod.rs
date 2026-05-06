pub mod types;
pub mod config;
pub mod kv_store;
pub mod runtime;
pub mod model;
pub mod backend;

pub use types::*;
pub use config::*;
pub use kv_store::*;
pub use runtime::*;
pub use model::*;
pub use backend::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::MemoryRuntimeConfig;
    use std::str::FromStr;

    #[test]
    fn test_quantization_from_str() {
        assert_eq!(Quantization::from_str("q4_0").unwrap(), Quantization::Q4_0);
        assert_eq!(Quantization::from_str("q4k").unwrap(), Quantization::Q4K);
        assert_eq!(Quantization::from_str("f32").unwrap(), Quantization::F32);
        assert!(Quantization::from_str("invalid").is_err());
    }

    #[test]
    fn test_metal_device_info_traits() {
        let info = MetalDeviceInfo {
            name: "M3".to_string(),
            unified_memory: true,
        };
        let info2 = info.clone();
        assert_eq!(info, info2);
        assert!(format!("{:?}", info).contains("M3"));
    }

    #[test]
    fn test_metal_error_variants() {
        let e1 = MetalError::DeviceNotFound;
        let e2 = MetalError::LibraryError("lib".into());
        let e3 = MetalError::PipelineError("pipe".into());
       
        assert_eq!(e1, e1.clone());
        assert_ne!(e1, e2);
       
        assert!(format!("{}", e1).contains("not found"));
        assert!(format!("{}", e2).contains("Metal library error: lib"));
        assert!(format!("{}", e3).contains("Metal pipeline error: pipe"));
    }

    #[test]
    fn test_backend_config_validation() {
        let mut cfg = MetalBackendConfig::default();
        cfg.max_context_tokens = 0;
        assert!(MetalBackend::new(cfg).is_err());

        let mut cfg = MetalBackendConfig::default();
        cfg.kv_bytes_per_token = 0;
        assert!(MetalBackend::new(cfg).is_err());
    }

    #[test]
    fn test_runtime_config_validation() {
        let base_cfg = MetalRuntimeConfig {
            model_name: "test".to_string(),
            memory: MemoryRuntimeConfig::cpu(4096, 16, 32, 32, 32),
            quantization: Quantization::F32,
        };

        let mut cfg = base_cfg.clone();
        cfg.model_name = "  ".to_string();
        assert!(MetalMemoryRuntime::new(cfg).is_err());

        let mut cfg = base_cfg.clone();
        cfg.memory.bytes_per_token = 0;
        assert!(MetalMemoryRuntime::new(cfg).is_err());
    }
}
