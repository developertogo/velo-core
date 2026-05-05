use std::sync::{Arc, Mutex};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use crate::backend::{CausalLmBackend, TokenLogits};
use crate::radix_cache::{CacheLookup, TokenId};
use crate::slot_manager::SlotId;
use crate::speculative::{Result, SpeculativeError};

use super::config::MetalBackendConfig;
use super::kv_store::SharedMetalKvStore;
use super::model::LlamaMetalModel;
use super::runtime::{MetalMemoryRuntime, SharedPagedAttentionBlockManager};
use super::types::MetalDeviceInfo;

/// A high-performance inference backend using Apple Metal.
#[derive(Debug, Clone)]
pub struct MetalBackend {
    pub config: MetalBackendConfig,
    pub device: MetalDeviceInfo,
    pub bound_prefix: Option<CacheLookup>,
    pub model: Option<Arc<Mutex<LlamaMetalModel>>>,
    pub allocator: Option<SharedPagedAttentionBlockManager>,
    pub store: Option<SharedMetalKvStore>,
    pub slot_id: Option<SlotId>,
    pub slot_mapping: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl MetalBackend {
    pub fn new(config: MetalBackendConfig) -> Result<Self> {
        if config.max_context_tokens == 0 {
            return Err(SpeculativeError::Model(
                "max_context_tokens must be greater than zero".to_string(),
            ));
        }
        if config.kv_bytes_per_token == 0 {
            return Err(SpeculativeError::Model(
                "kv_bytes_per_token must be greater than zero".to_string(),
            ));
        }

        Ok(Self {
            config,
            device: MetalDeviceInfo {
                name: "apple-metal-placeholder".to_string(),
                unified_memory: true,
            },
            bound_prefix: None,
            model: None,
            allocator: None,
            store: None,
            slot_id: None,
            slot_mapping: None,
        })
    }

    pub fn with_model(mut self, model: LlamaMetalModel) -> Self {
        self.model = Some(Arc::new(Mutex::new(model)));
        self
    }

    pub fn config(&self) -> &MetalBackendConfig {
        &self.config
    }

    pub fn device(&self) -> &MetalDeviceInfo {
        &self.device
    }

    pub fn bound_prefix(&self) -> Option<&CacheLookup> {
        self.bound_prefix.as_ref()
    }

    fn not_initialized() -> SpeculativeError {
        SpeculativeError::Model(
            "Metal backend skeleton is not wired to model weights or kernels yet".to_string(),
        )
    }

    pub fn wire(&mut self, store: crate::model_loader::WeightStore, runtime: &MetalMemoryRuntime) -> Result<()> {
        let handles = runtime.context().handles.clone();
        let device = handles.device.clone().ok_or_else(|| SpeculativeError::Model("Runtime missing device".into()))?;
        let queue = handles.command_queue.clone().ok_or_else(|| SpeculativeError::Model("Runtime missing queue".into()))?;
        let library = handles.library.clone().ok_or_else(|| SpeculativeError::Model("Runtime missing library".into()))?;

        let mut model = LlamaMetalModel::new(store.meta.clone(), device, queue, library);
        model.upload_weights(&store)?;
        
        self.model = Some(Arc::new(Mutex::new(model)));
        self.device = runtime.context().device.clone();
        self.allocator = Some(runtime.allocator.clone());
        self.store = Some(runtime.store.clone());
        self.slot_mapping = Some(runtime.slot_mapping.clone());
        
        Ok(())
    }
}

impl CausalLmBackend for MetalBackend {
    fn bind_prefix_cache(&mut self, prefix: &CacheLookup) -> Result<()> {
        self.bound_prefix = Some(prefix.clone());
        Ok(())
    }

    fn bind_slot(&mut self, slot: SlotId) -> Result<()> {
        self.slot_id = Some(slot);
        Ok(())
    }

    fn next_logits(&mut self, context: &[TokenId]) -> Result<TokenLogits> {
        let Some(model_arc) = &self.model else {
            return Err(Self::not_initialized());
        };
        let allocator_arc = self.allocator.as_ref().ok_or_else(|| SpeculativeError::Model("Allocator not wired".into()))?;
        let store_arc = self.store.as_ref().ok_or_else(|| SpeculativeError::Model("Store not wired".into()))?;

        let last = context.last().copied().ok_or_else(|| {
            SpeculativeError::Model("next_logits called with empty context".to_string())
        })?;

        let prefix = self.bound_prefix.as_ref().ok_or_else(|| SpeculativeError::Model("No prefix bound".to_string()))?;
        let prefix_handle = prefix.handle.ok_or_else(|| {
            SpeculativeError::Model("Prefix has no KV handle".to_string())
        })?;

        let mut model = model_arc.lock().unwrap();
        let allocator = allocator_arc.0.lock().unwrap();
        let store = store_arc.0.lock().unwrap();

        let span = allocator.materialize_span(prefix_handle).ok_or_else(|| {
            SpeculativeError::Model(format!("Could not materialize span for prefix {:?}", prefix_handle))
        })?;

        let page_bytes = self.config.paged_block_size * self.config.kv_bytes_per_token;
        let mut block_indices = Vec::with_capacity(span.pages.len());
        for page_id in span.pages {
            let block = store.get_block(crate::radix_cache::KvCacheHandle { block_id: page_id.0, token_len: 0 })
                .ok_or_else(|| SpeculativeError::Model(format!("Block {:?} not found in store", page_id)))?;
            block_indices.push((block.offset / page_bytes) as u32);
        }

        let pos = context.len() - 1; 
        let logits = model.forward_one(
            last,
            pos,
            store.k_pool(),
            store.v_pool(),
            self.slot_id.ok_or_else(|| SpeculativeError::Model("Slot not bound".into()))?,
            self.slot_mapping.as_ref().ok_or_else(|| SpeculativeError::Model("Slot mapping not wired".into()))?,
            allocator.config().total_pages,
            self.config.paged_block_size,
        )?;
        TokenLogits::new(logits)
    }

    fn verify_logits(
        &mut self,
        context: &[TokenId],
        drafted: &[TokenId],
    ) -> Result<Vec<TokenLogits>> {
        let Some(model_arc) = &self.model else {
            return Err(Self::not_initialized());
        };
        let allocator_arc = self.allocator.as_ref().ok_or_else(|| SpeculativeError::Model("Allocator not wired".into()))?;
        let store_arc = self.store.as_ref().ok_or_else(|| SpeculativeError::Model("Store not wired".into()))?;

        let prefix = self.bound_prefix.as_ref().ok_or_else(|| SpeculativeError::Model("No prefix bound".to_string()))?;
        let prefix_handle = prefix.handle.ok_or_else(|| {
            SpeculativeError::Model("Prefix has no KV handle".to_string())
        })?;

        let mut model = model_arc.lock().unwrap();
        let allocator = allocator_arc.0.lock().unwrap();
        let store = store_arc.0.lock().unwrap();

        let span = allocator.materialize_span(prefix_handle).ok_or_else(|| {
            SpeculativeError::Model(format!("Could not materialize span for prefix {:?}", prefix_handle))
        })?;

        let page_bytes = self.config.paged_block_size * self.config.kv_bytes_per_token;
        let mut block_indices = Vec::with_capacity(span.pages.len());
        for page_id in span.pages {
            let block = store.get_block(crate::radix_cache::KvCacheHandle { block_id: page_id.0, token_len: 0 })
                .ok_or_else(|| SpeculativeError::Model(format!("Block {:?} not found in store", page_id)))?;
            block_indices.push((block.offset / page_bytes) as u32);
        }

        let mut result = Vec::with_capacity(drafted.len());
        let mut current_pos = context.len();
        
        let slot_id = self.slot_id.ok_or_else(|| SpeculativeError::Model("Slot not bound".into()))?;
        let slot_mapping = self.slot_mapping.as_ref().ok_or_else(|| SpeculativeError::Model("Slot mapping not wired".into()))?;
        let max_pages = allocator.config().total_pages;

        for &tok in drafted {
            let logits = model.forward_one(
                tok,
                current_pos,
                store.k_pool(),
                store.v_pool(),
                slot_id,
                slot_mapping,
                max_pages,
                self.config.paged_block_size,
            )?;
            result.push(TokenLogits::new(logits)?);
            current_pos += 1;
        }
        Ok(result)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_backend_new_validation() {
        let mut cfg = MetalBackendConfig::default();
        cfg.max_context_tokens = 0;
        assert!(MetalBackend::new(cfg).is_err());
        
        let mut cfg = MetalBackendConfig::default();
        cfg.kv_bytes_per_token = 0;
        assert!(MetalBackend::new(cfg).is_err());
    }

    #[test]
    fn test_metal_backend_not_initialized() {
        let backend = MetalBackend::new(MetalBackendConfig::default()).unwrap();
        assert!(format!("{:?}", backend).contains("MetalBackend"));
    }
}
