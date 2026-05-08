use std::collections::HashMap;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

/// Unique identifier for a LoRA adapter.
pub type AdapterId = u32;

/// Configuration for a Low-Rank Adaptation (LoRA) adapter.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// The rank of the adaptation matrices.
    pub r: usize,
    /// The scaling factor (alpha) applied to the LoRA delta.
    pub alpha: f32,
    /// List of model modules (e.g. "attention.wq") targetted by this adapter.
    pub target_modules: Vec<String>,
}

/// GPU-resident weights for a specific LoRA adapter.
pub struct LoraWeights {
    /// Map of tensor names to Metal buffers (e.g. "layers.0.attention.wq.lora_A.weight").
    pub weights: HashMap<String, Retained<ProtocolObject<dyn MTLBuffer>>>,
}

/// Registry for managing multiple LoRA adapters in memory.
pub struct LoraRegistry {
    pub adapters: HashMap<AdapterId, LoraWeights>,
    pub configs: HashMap<AdapterId, LoraConfig>,
}

impl LoraRegistry {
    /// Creates a new, empty LoRA registry.
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            configs: HashMap::new(),
        }
    }

    /// Adds a new adapter to the registry.
    pub fn add_adapter(&mut self, id: AdapterId, weights: LoraWeights, config: LoraConfig) {
        self.adapters.insert(id, weights);
        self.configs.insert(id, config);
    }

    /// Retrieves the weights for a specific adapter ID.
    pub fn get_weights(&self, id: AdapterId) -> Option<&LoraWeights> {
        self.adapters.get(&id)
    }

    /// Retrieves the configuration for a specific adapter ID.
    pub fn get_config(&self, id: AdapterId) -> Option<&LoraConfig> {
        self.configs.get(&id)
    }

    /// Hints the registry to ensure the specified adapters are resident in memory.
    pub fn prefetch(&mut self, ids: &[AdapterId]) {
        for id in ids {
            if !self.adapters.contains_key(id) {
                println!("Prefetching adapter {}...", id);
            }
        }
    }

    /// Removes adapters that are not in the provided active list.
    pub fn evict_unused(&mut self, active_ids: &[AdapterId]) {
        let active: std::collections::HashSet<_> = active_ids.iter().collect();
        self.adapters.retain(|id, _| active.contains(id));
        self.configs.retain(|id, _| active.contains(id));
    }
}

impl LoraWeights {
    /// Creates LoRA weights from raw byte slices and uploads them to the Metal device.
    pub fn from_raw(
        device: &ProtocolObject<dyn MTLDevice>,
        weights: HashMap<String, Vec<u8>>,
    ) -> Self {
        let mut mtl_weights = HashMap::new();
        for (name, data) in weights {
            let buffer = unsafe {
                device.newBufferWithBytes_length_options(
                    std::ptr::NonNull::new(data.as_ptr() as *mut _).unwrap(),
                    data.len() as _,
                    objc2_metal::MTLResourceOptions::StorageModeShared,
                ).unwrap()
            };
            mtl_weights.insert(name, buffer);
        }
        Self { weights: mtl_weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_registry_management() {
        let mut registry = LoraRegistry::new();
        let config = LoraConfig {
            r: 8,
            alpha: 16.0,
            target_modules: vec!["attention.wq".to_string()],
        };
        
        // Mock weights (empty because we can't create MTLBuffer easily in unit test without device)
        let weights = LoraWeights { weights: HashMap::new() };
        
        registry.add_adapter(1, weights, config.clone());
        
        assert!(registry.get_config(1).is_some());
        assert_eq!(registry.get_config(1).unwrap().r, 8);
        assert!(registry.get_weights(1).is_some());
        
        registry.prefetch(&[1, 2]); // Should print prefetch for 2
        
        registry.evict_unused(&[2]);
        assert!(registry.get_config(1).is_none());
    }
}
