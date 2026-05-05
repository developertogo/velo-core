use crate::gguf::GgmlType;

/// Quantization formats supported by the Metal backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    /// 4-bit quantization (block-based).
    Q4_0,
    /// 4-bit K-quantization (Super-block).
    Q4K,
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point.
    F16,
    /// 8-bit quantization.
    Q8K,
}

impl std::str::FromStr for Quantization {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "Q4_0" => Ok(Self::Q4_0),
            "Q4K" | "Q4_K" => Ok(Self::Q4K),
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "Q8K" | "Q8_K" => Ok(Self::Q8K),
            _ => Err(format!("unknown quantization: {}", s)),
        }
    }
}

impl Quantization {
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4K | Self::Q8K => 32,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4K => "Q4_K",
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q8K => "Q8_K",
        }
    }
    
    pub fn from_ggml(dtype: GgmlType) -> Option<Self> {
        match dtype {
            GgmlType::F32 => Some(Self::F32),
            GgmlType::F16 => Some(Self::F16),
            GgmlType::Q4_0 => Some(Self::Q4_0),
            GgmlType::Q4K => Some(Self::Q4K),
            GgmlType::Q8K => Some(Self::Q8K),
            _ => None,
        }
    }
}

/// Placement strategy for Metal buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalBufferPlacement {
    /// Unified memory (shared between CPU and GPU). Default for Apple Silicon.
    Unified,
    /// GPU-private memory.
    Private,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalError {
    DeviceNotFound,
    LibraryError(String),
    PipelineError(String),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceNotFound => write!(f, "Metal device not found"),
            Self::LibraryError(s) => write!(f, "Metal library error: {}", s),
            Self::PipelineError(s) => write!(f, "Metal pipeline error: {}", s),
        }
    }
}

impl std::error::Error for MetalError {}

/// Information about the Metal hardware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    /// Human-readable name of the GPU (e.g., "Apple M3 Max").
    pub name: String,
    /// Whether the device supports unified memory architecture.
    pub unified_memory: bool,
}
