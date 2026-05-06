use crate::runtime::MemoryRuntimeConfig;
use super::types::Quantization;

/// Configuration for the Metal backend.
#[derive(Debug, Clone)]
pub struct MetalBackendConfig {
    /// Name of the model.
    pub model_name: String,
    /// Maximum context length in tokens.
    pub max_context_tokens: usize,
    /// Number of bytes required per KV token.
    pub kv_bytes_per_token: usize,
    /// Number of tokens per page/block.
    pub paged_block_size: usize,
    /// Quantization format used by the weights.
    pub quantization: Quantization,
}

impl Default for MetalBackendConfig {
    fn default() -> Self {
        Self {
            model_name: "llama-metal".to_string(),
            max_context_tokens: 4096,
            kv_bytes_per_token: 4,
            paged_block_size: 16,
            quantization: Quantization::Q4_0,
        }
    }
}

/// Configuration for the Metal memory runtime.
#[derive(Debug, Clone)]
pub struct MetalRuntimeConfig {
    /// Name of the model.
    pub model_name: String,
    /// Memory allocation configuration.
    pub memory: MemoryRuntimeConfig,
    /// Quantization format.
    pub quantization: Quantization,
}

impl Default for MetalRuntimeConfig {
    fn default() -> Self {
        Self {
            model_name: "llama-metal".to_string(),
            memory: MemoryRuntimeConfig::default(),
            quantization: Quantization::Q4_0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::Quantization;

    #[test]
    fn test_metal_backend_config_default() {
        let cfg = MetalBackendConfig::default();
        assert_eq!(cfg.model_name, "llama-metal");
        assert_eq!(cfg.quantization, Quantization::Q4_0);
    }

    #[test]
    fn test_metal_runtime_config_default() {
        let cfg = MetalRuntimeConfig::default();
        assert_eq!(cfg.model_name, "llama-metal");
        assert_eq!(cfg.quantization, Quantization::Q4_0);
    }
}
