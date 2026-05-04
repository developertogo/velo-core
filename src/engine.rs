use crate::kv_store::KvStoreError;
use crate::paged_attention::{PageManagerError, PageSpan};
use crate::radix_cache::{CacheLookup, KvCacheHandle, RadixCache, TokenId};
use crate::runtime::{
    CpuMemoryRuntime, KvBlockStore, MemoryRuntime, MemoryRuntimeConfig, PagedBlockAllocator,
};
use crate::speculative::{
    DraftModel, SpeculativeDecoder, SpeculativeError, SpeculativeStats, TargetModel,
};

/// High-performance inference engine for speculative decoding.
///
/// `VeloEngine` orchestrates the radix-prefix cache, paged-attention memory,
/// and the speculative draft/verify loop. It uses a `SlotPool` to isolate
/// concurrent requests and provide stable indexing for GPU backends.
#[derive(Debug)]
pub struct VeloEngine<R = CpuMemoryRuntime> {
    /// Orchestrates the draft-and-verify speculative loop.
    decoder: SpeculativeDecoder,
    /// Manages prefix KV-cache reuse and eviction.
    prefix_cache: RadixCache,
    /// Backend-specific memory and execution runtime (CPU or Metal).
    runtime: R,
    /// Pool of stable indexes for concurrent request state.
    slot_pool: crate::slot_manager::SlotPool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    pub draft_window: usize,
    pub memory: MemoryRuntimeConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineOutput {
    pub tokens: Vec<TokenId>,
    pub cached_prefix: CacheLookup,
    pub cached_pages: Option<PageSpan>,
    pub inserted_handle: Option<KvCacheHandle>,
    pub inserted_pages: Option<PageSpan>,
    pub stats: EngineStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnginePrefillOutput {
    pub cached_prefix: CacheLookup,
    pub cached_pages: Option<PageSpan>,
    pub inserted_handle: Option<KvCacheHandle>,
    pub inserted_pages: Option<PageSpan>,
    pub stats: EngineStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EngineStats {
    pub cache_hit_tokens: usize,
    pub cache_miss_tokens: usize,
    pub speculative: SpeculativeStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    KvStore(KvStoreError),
    PagedAttention(PageManagerError),
    Speculative(SpeculativeError),
}

impl VeloEngine {
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        let runtime = CpuMemoryRuntime::new(config.memory)?;

        Self::with_runtime(config, runtime)
    }
}

impl<R> VeloEngine<R>
where
    R: MemoryRuntime,
{
    pub fn with_runtime(config: EngineConfig, runtime: R) -> Result<Self, EngineError> {
        Ok(Self {
            decoder: SpeculativeDecoder::new(config.draft_window)?,
            prefix_cache: RadixCache::new(),
            slot_pool: crate::slot_manager::SlotPool::new(config.memory.max_slots),
            runtime,
        })
    }

    pub fn decoder(&self) -> &SpeculativeDecoder {
        &self.decoder
    }

    pub fn generate<D, T>(
        &mut self,
        draft_model: &mut D,
        target_model: &mut T,
        prompt: &[TokenId],
        max_new_tokens: usize,
    ) -> Result<EngineOutput, EngineError>
    where
        D: DraftModel,
        T: TargetModel,
    {
        let slot_id = self.slot_pool.alloc().ok_or_else(|| {
            EngineError::Speculative(SpeculativeError::Model("No free slots available".to_string()))
        })?;

        let prefill = self.prefill(prompt)?;

        // Materialize the full page sequence for the slot
        let pages = prefill.inserted_pages.as_ref()
            .or(prefill.cached_pages.as_ref())
            .ok_or_else(|| EngineError::Speculative(SpeculativeError::Model("Failed to resolve pages for prefill".to_string())))?;
        
        let page_ids: Vec<u32> = pages.pages.iter().map(|p| p.0 as u32).collect();
        self.runtime.bind_slot(slot_id, &page_ids)?;

        draft_model.bind_slot(slot_id)?;
        target_model.bind_slot(slot_id)?;
        draft_model.bind_prefix_cache(&prefill.cached_prefix)?;
        target_model.bind_prefix_cache(&prefill.cached_prefix)?;

        let decoded =
            self.decoder
                .generate(draft_model, target_model, prompt, max_new_tokens)?;

        self.slot_pool.release(slot_id);

        let mut full_sequence = Vec::with_capacity(prompt.len() + decoded.tokens.len());
        full_sequence.extend_from_slice(prompt);
        full_sequence.extend_from_slice(&decoded.tokens);

        let inserted_handle = (!full_sequence.is_empty()).then(|| -> Result<_, EngineError> {
            self.cache_sequence(&full_sequence)
        }).transpose()?;

        let cached_pages = prefill
            .cached_prefix
            .handle
            .and_then(|handle| self.runtime.allocator().materialize_span(handle));

        Ok(EngineOutput {
            tokens: decoded.tokens,
            cached_prefix: prefill.cached_prefix,
            cached_pages,
            inserted_handle,
            inserted_pages: inserted_handle
                .and_then(|handle| self.runtime.allocator().materialize_span(handle)),
            stats: EngineStats {
                cache_hit_tokens: prefill.stats.cache_hit_tokens,
                cache_miss_tokens: prefill.stats.cache_miss_tokens,
                speculative: decoded.stats,
            },
        })
    }

    pub fn prefill(&mut self, prompt: &[TokenId]) -> Result<EnginePrefillOutput, EngineError> {
        let cached_prefix = self.prefix_cache.lookup(prompt);
        let inserted_handle = if !prompt.is_empty() && cached_prefix.matched_tokens < prompt.len() {
            Some(self.cache_sequence(prompt)?)
        } else {
            None
        };

        Ok(EnginePrefillOutput {
            cached_pages: cached_prefix
                .handle
                .and_then(|handle| self.runtime.allocator().materialize_span(handle)),
            inserted_pages: inserted_handle
                .and_then(|handle| self.runtime.allocator().materialize_span(handle)),
            stats: EngineStats {
                cache_hit_tokens: cached_prefix.matched_tokens,
                cache_miss_tokens: prompt.len() - cached_prefix.matched_tokens,
                speculative: SpeculativeStats::default(),
            },
            cached_prefix,
            inserted_handle,
        })
    }

    pub fn cache(&self) -> &RadixCache {
        &self.prefix_cache
    }

    pub fn cache_mut(&mut self) -> &mut RadixCache {
        &mut self.prefix_cache
    }

    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut R {
        &mut self.runtime
    }

    fn cache_sequence(&mut self, tokens: &[TokenId]) -> Result<KvCacheHandle, EngineError> {
        let handle = self.runtime.store_mut().allocate(tokens.len())?;
        if let Err(error) = self.runtime.allocator_mut().allocate(handle, tokens.len()) {
            self.runtime.store_mut().release(handle)?;
            return Err(error.into());
        }

        let inserted = self.prefix_cache.insert(tokens, handle);
        if let Some(replaced) = inserted.replaced {
            self.runtime.store_mut().release(replaced)?;
            self.runtime.allocator_mut().release(replaced)?;
        }
        Ok(handle)
    }
}

impl From<KvStoreError> for EngineError {
    fn from(error: KvStoreError) -> Self {
        Self::KvStore(error)
    }
}

impl From<PageManagerError> for EngineError {
    fn from(error: PageManagerError) -> Self {
        Self::PagedAttention(error)
    }
}

impl From<SpeculativeError> for EngineError {
    fn from(error: SpeculativeError) -> Self {
        Self::Speculative(error)
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KvStore(error) => write!(formatter, "KV store failed: {error}"),
            Self::PagedAttention(error) => write!(formatter, "paged attention failed: {error}"),
            Self::Speculative(error) => write!(formatter, "speculative decoding failed: {error}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl VeloEngine<CpuMemoryRuntime> {
    pub fn kv_store(&self) -> &crate::kv_store::InMemoryKvStore {
        self.runtime.store()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_store::KvStore;
    use crate::metal::Quantization;
    use crate::{MetalMemoryRuntime, MetalRuntimeConfig};
    use crate::speculative::{NextTokenPrediction, Result as SpeculativeResult, VerifyStep};

    #[derive(Debug)]
    struct ScriptedDraft {
        script: Vec<TokenId>,
        bound_prefixes: Vec<CacheLookup>,
        bound_slots: Vec<crate::slot_manager::SlotId>,
    }

    impl DraftModel for ScriptedDraft {
        fn bind_prefix_cache(&mut self, prefix: &CacheLookup) -> SpeculativeResult<()> {
            self.bound_prefixes.push(prefix.clone());
            Ok(())
        }

        fn bind_slot(&mut self, slot: crate::slot_manager::SlotId) -> SpeculativeResult<()> {
            self.bound_slots.push(slot);
            Ok(())
        }

        fn draft(
            &mut self,
            context: &[TokenId],
            max_tokens: usize,
        ) -> SpeculativeResult<Vec<NextTokenPrediction>> {
            Ok(self.script[context.len()..]
                .iter()
                .take(max_tokens)
                .map(|token| NextTokenPrediction {
                    token: *token,
                    confidence: 1.0,
                })
                .collect())
        }
    }

    #[derive(Debug)]
    struct ScriptedTarget {
        script: Vec<TokenId>,
        bound_prefixes: Vec<CacheLookup>,
        bound_slots: Vec<crate::slot_manager::SlotId>,
    }

    impl TargetModel for ScriptedTarget {
        fn bind_prefix_cache(&mut self, prefix: &CacheLookup) -> SpeculativeResult<()> {
            self.bound_prefixes.push(prefix.clone());
            Ok(())
        }

        fn bind_slot(&mut self, slot: crate::slot_manager::SlotId) -> SpeculativeResult<()> {
            self.bound_slots.push(slot);
            Ok(())
        }

        fn verify(
            &mut self,
            context: &[TokenId],
            drafted: &[TokenId],
        ) -> SpeculativeResult<Vec<VerifyStep>> {
            Ok(self.script[context.len()..]
                .iter()
                .take(drafted.len())
                .map(|token| VerifyStep { expected: *token })
                .collect())
        }
    }

    #[test]
    fn generates_and_caches_full_sequence() {
        let mut engine = VeloEngine::new(EngineConfig {
            draft_window: 4,
            memory: MemoryRuntimeConfig::cpu(128, 16, 32, 32, 32),
        })
        .unwrap();
        let prompt = [10, 20];
        let mut draft = ScriptedDraft {
            script: vec![10, 20, 30, 40],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let mut target = ScriptedTarget {
            script: vec![10, 20, 30, 40],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };

        let output = engine
            .generate(&mut draft, &mut target, &prompt, 2)
            .unwrap();

        assert_eq!(output.tokens, vec![30, 40]);
        assert_eq!(output.cached_prefix.matched_tokens, 0);
        assert!(output.inserted_pages.is_some());
        assert_eq!(
            output.inserted_handle,
            Some(KvCacheHandle {
                block_id: 2,
                token_len: 4,
            })
        );
        assert_eq!(
            engine.cache_mut().lookup(&[10, 20, 30, 40]).matched_tokens,
            4
        );
        assert_eq!(engine.kv_store().allocated_bytes(), 768);
    }

    #[test]
    fn reports_cache_hit_for_repeated_prompt_prefix() {
        let mut engine = VeloEngine::new(EngineConfig {
            draft_window: 2,
            memory: MemoryRuntimeConfig::cpu(128, 16, 32, 32, 32),
        })
        .unwrap();
        let mut draft = ScriptedDraft {
            script: vec![1, 2, 3, 4, 5],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3, 4, 5],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };

        engine.generate(&mut draft, &mut target, &[1, 2], 2).unwrap();

        let mut draft = ScriptedDraft {
            script: vec![1, 2, 3, 4, 5],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3, 4, 5],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let output = engine
            .generate(&mut draft, &mut target, &[1, 2, 3, 4], 1)
            .unwrap();

        assert_eq!(output.cached_prefix.matched_tokens, 4);
        assert!(output.cached_pages.is_some());
        assert_eq!(output.stats.cache_hit_tokens, 4);
        assert_eq!(output.stats.cache_miss_tokens, 0);
        assert_eq!(output.tokens, vec![5]);
    }

    #[test]
    fn releases_replaced_kv_block_for_same_sequence() {
        let mut engine = VeloEngine::new(EngineConfig {
            draft_window: 2,
            memory: MemoryRuntimeConfig::cpu(32, 8, 16, 32, 32),
        })
        .unwrap();
        let mut draft = ScriptedDraft {
            script: vec![1, 2, 3],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };

        let first = engine.generate(&mut draft, &mut target, &[1], 2).unwrap();

        let mut draft = ScriptedDraft {
            script: vec![1, 2, 3],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let second = engine.generate(&mut draft, &mut target, &[1], 2).unwrap();

        assert_eq!(first.inserted_handle.unwrap().block_id, 2);
        assert_eq!(second.inserted_handle.unwrap().block_id, 3);
        assert_eq!(engine.kv_store().len(), 2);
        assert!(engine.kv_store().get(second.inserted_handle.unwrap()).is_some());
    }

    #[test]
    fn binds_cached_prefix_to_both_models_before_generation() {
        let mut engine = VeloEngine::new(EngineConfig {
            draft_window: 2,
            memory: MemoryRuntimeConfig::cpu(16, 8, 16, 32, 32),
        })
        .unwrap();
        let mut draft = ScriptedDraft {
            script: vec![7, 8, 9],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let mut target = ScriptedTarget {
            script: vec![7, 8, 9],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };

        engine.generate(&mut draft, &mut target, &[7], 2).unwrap();

        let mut draft = ScriptedDraft {
            script: vec![7, 8, 9, 10],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let mut target = ScriptedTarget {
            script: vec![7, 8, 9, 10],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };

        engine
            .generate(&mut draft, &mut target, &[7, 8, 9], 1)
            .unwrap();

        assert_eq!(draft.bound_prefixes.len(), 1);
        assert_eq!(target.bound_prefixes.len(), 1);
        assert_eq!(draft.bound_prefixes[0].matched_tokens, 3);
        assert_eq!(target.bound_prefixes[0].matched_tokens, 3);
        assert!(draft.bound_prefixes[0].handle.is_some());
        assert_eq!(draft.bound_prefixes[0], target.bound_prefixes[0]);
    }

    #[test]
    fn can_generate_over_placeholder_metal_runtime() {
        let Ok(runtime) = MetalMemoryRuntime::new(MetalRuntimeConfig {
            model_name: "draft-1b".to_string(),
            memory: MemoryRuntimeConfig::cpu(32, 8, 16, 32, 32),
            quantization: Quantization::Q4K,
        }) else {
            // Skip if Metal device not found in test environment
            return;
        };
        let mut engine = VeloEngine::with_runtime(
            EngineConfig {
                draft_window: 2,
                memory: runtime.context().memory,
            },
            runtime,
        )
        .unwrap();
        let mut draft = ScriptedDraft {
            script: vec![1, 2, 3, 4],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3, 4],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };

        let output = engine.generate(&mut draft, &mut target, &[1], 2).unwrap();

        assert_eq!(output.tokens, vec![2, 3]);
        assert_eq!(engine.runtime().context().placement, crate::metal::MetalBufferPlacement::Unified);
        assert_eq!(engine.runtime().store().allocated_bytes(), 128);
        assert_eq!(draft.bound_slots.len(), 1);
        assert_eq!(target.bound_slots.len(), 1);
    }

    #[test]
    fn test_slot_exhaustion() {
        let config = EngineConfig {
            draft_window: 4,
            memory: MemoryRuntimeConfig::cpu(128, 16, 32, 1, 1), // Only 1 slot
        };
        let mut engine = VeloEngine::new(config).unwrap();
        let mut draft = ScriptedDraft {
            script: vec![1, 2, 3, 4],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3, 4],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };

        // First one works
        let _ = engine.generate(&mut draft, &mut target, &[1], 2).unwrap();
        
        // Second one also works because the slot was released
        let _ = engine.generate(&mut draft, &mut target, &[1], 2).unwrap();
        
        assert_eq!(draft.bound_slots.len(), 2);
    }
}
