use crate::kv_store::KvStoreError;
use crate::paged_attention::{PageManagerError, PageSpan, KvCacheType};
use crate::radix_cache::{CacheLookup, KvCacheHandle, RadixCache, TokenId};
use crate::runtime::{
    CpuMemoryRuntime, KvBlockStore, MemoryRuntime, MemoryRuntimeConfig, PagedBlockAllocator,
};
use crate::speculative::{
    SpeculativeDecoder, SpeculativeError, SpeculativeStats,
};

/// High-performance inference engine for speculative decoding.
///
/// `VeloEngine` orchestrates the radix-prefix cache, paged-attention memory,
/// and the speculative draft/verify loop. It uses a `SlotPool` to isolate
/// concurrent requests and provide stable indexing for GPU backends.
pub struct VeloEngine<R = CpuMemoryRuntime> {
    /// Orchestrates the draft-and-verify speculative loop.
    decoder: SpeculativeDecoder,
    /// Manages prefix KV-cache reuse and eviction.
    prefix_cache: RadixCache,
    /// Backend-specific memory and execution runtime (CPU or Metal).
    runtime: R,
    /// Pool of stable indexes for concurrent request state.
    slot_pool: crate::slot_manager::SlotPool,
    /// Factory for creating grammar matchers.
    pub parser_factory: Option<std::sync::Arc<llguidance::ParserFactory>>,
}

impl<R> std::fmt::Debug for VeloEngine<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VeloEngine")
            .field("decoder", &self.decoder)
            .field("prefix_cache", &self.prefix_cache)
            .field("slot_pool", &self.slot_pool)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    pub draft_window: usize,
    pub memory: MemoryRuntimeConfig,
    pub kv_type: KvCacheType,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            draft_window: 8,
            memory: MemoryRuntimeConfig::default(),
            kv_type: KvCacheType::Fp32,
        }
    }
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

/// A request to be processed in a batch.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// The prompt tokens.
    pub prompt: Vec<TokenId>,
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Optional grammar/regex constraint.
    pub constraint: Option<crate::constraints::Constraint>,
}

/// Internal state for a request currently being processed in a batch.
pub struct ActiveRequest {
    /// The speculative decoding session.
    pub session: crate::speculative::SpeculativeSession,
    /// The GPU slot assigned to this request.
    pub slot_id: crate::slot_manager::SlotId,
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Tokens generated so far.
    pub generated: Vec<TokenId>,
    /// Prefill results (prefix cache hits/misses).
    pub prefill: EnginePrefillOutput,
    /// Optional grammar matcher for constrained decoding.
    pub matcher: Option<Box<dyn crate::constraints::CfgMatcher>>,
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
            parser_factory: None,
        })
    }

    pub fn decoder(&self) -> &SpeculativeDecoder {
        &self.decoder
    }

    pub fn slot_pool_capacity(&self) -> usize {
        self.slot_pool.capacity()
    }

    pub fn allocate_slot(&mut self) -> Option<crate::slot_manager::SlotId> {
        self.slot_pool.alloc()
    }

    pub fn release_slot(&mut self, slot_id: crate::slot_manager::SlotId) {
        self.slot_pool.release(slot_id);
    }

    pub fn generate<D, T>(
        &mut self,
        draft_model: &mut D,
        target_model: &mut T,
        prompt: &[TokenId],
        max_new_tokens: usize,
    ) -> Result<EngineOutput, EngineError>
    where
        D: crate::speculative::DraftModel,
        T: crate::speculative::TargetModel,
    {
        let outputs = self.generate_batch(
            draft_model,
            target_model,
            vec![BatchRequest {
                prompt: prompt.to_vec(),
                max_new_tokens,
                constraint: None,
            }],
        )?;

        Ok(outputs.into_iter().next().unwrap())
    }

    /// Generates tokens for multiple requests concurrently using speculative decoding.
    ///
    /// This method orchestrates the draft-and-verify loop for all requests in the batch.
    /// It leverages the `SlotPool` for memory isolation and handles asynchronous finishing
    /// (requests can have different `max_new_tokens` or finish early).
    pub fn generate_batch<D, T>(
        &mut self,
        draft_model: &mut D,
        target_model: &mut T,
        requests: Vec<BatchRequest>,
    ) -> Result<Vec<EngineOutput>, EngineError>
    where
        D: crate::speculative::DraftModel,
        T: crate::speculative::TargetModel,
    {
        let mut active = Vec::with_capacity(requests.len());

        for req in requests {
            let slot_id = self.slot_pool.alloc().ok_or_else(|| {
                EngineError::Speculative(crate::speculative::SpeculativeError::Model(
                    "No free slots available".to_string(),
                ))
            })?;

            let prefill = self.prefill(&req.prompt)?;
            let pages = prefill
                .inserted_pages
                .as_ref()
                .or(prefill.cached_pages.as_ref())
                .ok_or_else(|| {
                    EngineError::Speculative(crate::speculative::SpeculativeError::Model(
                        "Failed to resolve pages for prefill".to_string(),
                    ))
                })?;

            let page_ids: Vec<u32> = pages.pages.iter().map(|p| p.0 as u32).collect();
            self.runtime.bind_slot(slot_id, &page_ids)?;

            let session = self.decoder.begin(&req.prompt)?;

            let mut matcher = if let Some(constraint) = req.constraint {
                 if let Some(factory) = &self.parser_factory {
                    let grammar = match constraint {
                        crate::constraints::Constraint::Regex(r) => llguidance::api::TopLevelGrammar::from_regex(&r),
                        crate::constraints::Constraint::JsonSchema(j) => llguidance::api::TopLevelGrammar::from_json_schema(j),
                        crate::constraints::Constraint::Lark(l) => llguidance::api::TopLevelGrammar::from_lark(l),
                    };
                    let vocab_size = factory.tok_env().tok_trie().vocab_size() as usize;
                    match crate::constraints::LlguidanceMatcher::new(factory, grammar, vocab_size) {
                        Ok(m) => Some(Box::new(m) as Box<dyn crate::constraints::CfgMatcher>),
                        Err(e) => {
                            eprintln!("Failed to create LlguidanceMatcher: {}", e);
                            None
                        }
                    }
                 } else {
                    eprintln!("No parser_factory found in engine");
                    None
                 }
            } else {
                None
            };

            if let Some(m) = matcher.as_mut() {
                for &token in &req.prompt {
                    m.advance(token);
                }
            }

            active.push(ActiveRequest {
                session,
                slot_id,
                max_new_tokens: req.max_new_tokens,
                generated: Vec::new(),
                prefill,
                matcher,
            });
        }

        // Bind prefix caches before starting
        for req in &mut active {
            draft_model.bind_slot(req.slot_id)?;
            target_model.bind_slot(req.slot_id)?;
            draft_model.bind_prefix_cache(&req.prefill.cached_prefix)?;
            target_model.bind_prefix_cache(&req.prefill.cached_prefix)?;
        }

        while active.iter().any(|req| req.generated.len() < req.max_new_tokens) {
            let active_indices: Vec<usize> = active
                .iter()
                .enumerate()
                .filter(|(_, req)| req.generated.len() < req.max_new_tokens)
                .map(|(i, _)| i)
                .collect();

            if active_indices.is_empty() {
                break;
            }

            // 1. Prepare and execute Draft Batch
            let draft_results = {
                let mut draft_matchers: Vec<Option<Box<dyn crate::constraints::CfgMatcher>>> = active
                    .iter()
                    .map(|req| req.matcher.as_ref().map(|m| m.clone_box()))
                    .collect();

                let mut draft_reqs: Vec<(&[TokenId], usize, Option<&mut (dyn crate::constraints::CfgMatcher + '_)>)> = active
                    .iter_mut()
                    .zip(draft_matchers.iter_mut())
                    .filter(|(req, _)| req.generated.len() < req.max_new_tokens)
                    .map(|(req, matcher)| {
                        let remaining = req.max_new_tokens - req.generated.len();
                        let requested = remaining.min(self.decoder.draft_window());
                        (req.session.context(), requested, matcher.as_deref_mut())
                    })
                    .collect();

                draft_model.draft_batch(&mut draft_reqs)?
            };

            for &idx in &active_indices {
                active[idx].session.record_draft_call();
            }

            // 2. Prepare and execute Verify Batch
            let mut drafted_tokens_storage = Vec::with_capacity(active_indices.len());
            for i in 0..active_indices.len() {
                let drafted_tokens: Vec<TokenId> =
                    draft_results[i].iter().map(|p| p.token).collect();
                drafted_tokens_storage.push(drafted_tokens);
            }

            let verify_results = {
                let mut verify_reqs: Vec<(&[TokenId], &[TokenId], Option<&mut (dyn crate::constraints::CfgMatcher + '_)>)> = active
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, req)| req.generated.len() < req.max_new_tokens)
                    .enumerate()
                    .map(|(batch_idx, (_, req))| {
                        (req.session.context(), drafted_tokens_storage[batch_idx].as_slice(), req.matcher.as_deref_mut())
                    })
                    .collect();

                target_model.verify_batch(&mut verify_reqs)?
            };

            for &idx in &active_indices {
                active[idx].session.record_target_call();
            }

            // 3. Commit results
            for (i, &idx) in active_indices.iter().enumerate() {
                let req = &mut active[idx];
                let accepted = req.session.commit(&draft_results[i], &verify_results[i])?;
                req.generated.extend_from_slice(&accepted);

                if req.session.has_pending_rejection() {
                    if let Some(token) = req.session.take_rejected_token() {
                        req.generated.push(token);
                    }
                }
            }
        }

        let mut final_outputs = Vec::with_capacity(active.len());
        for req in active {
            self.slot_pool.release(req.slot_id);

            let mut full_sequence = Vec::with_capacity(req.session.prompt().len() + req.generated.len());
            full_sequence.extend_from_slice(req.session.prompt());
            full_sequence.extend_from_slice(&req.generated);

            let inserted_handle = if !full_sequence.is_empty() {
                Some(self.cache_sequence(&full_sequence)?)
            } else {
                None
            };

            final_outputs.push(EngineOutput {
                tokens: req.generated,
                cached_prefix: req.prefill.cached_prefix,
                cached_pages: req.prefill.cached_pages,
                inserted_handle,
                inserted_pages: inserted_handle
                    .and_then(|handle| self.runtime.allocator().materialize_span(handle)),
                stats: EngineStats {
                    cache_hit_tokens: req.prefill.stats.cache_hit_tokens,
                    cache_miss_tokens: req.prefill.stats.cache_miss_tokens,
                    speculative: req.session.stats().clone(),
                },
            });
        }

        Ok(final_outputs)
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
    use crate::radix_cache::KvCacheHandle;
    use crate::paged_attention::KvCacheType;
    use crate::kv_store::KvStore;
    use crate::metal::Quantization;
    use crate::{MetalMemoryRuntime, MetalRuntimeConfig};
    use crate::speculative::{
        DraftModel, NextTokenPrediction, Result as SpeculativeResult, TargetModel, VerifyStep,
    };

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
            _matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>,
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
            _matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>,
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
            kv_type: KvCacheType::Fp32,
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
            kv_type: KvCacheType::Fp32,
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
            kv_type: KvCacheType::Fp32,
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
            kv_type: KvCacheType::Fp32,
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
                kv_type: KvCacheType::Fp32,
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
            kv_type: KvCacheType::Fp32,
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

    #[test]
    fn test_multi_request_batching() {
        let config = EngineConfig {
            draft_window: 2,
            memory: MemoryRuntimeConfig::cpu(1024, 16, 32, 4, 4), // 4 slots
            kv_type: KvCacheType::Fp32,
        };
        let mut engine = VeloEngine::new(config).unwrap();

        // Req 1: generates [2, 3]
        // Req 2: generates [11, 12, 13]
        let mut draft = ScriptedDraft {
            script: vec![0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };

        let mut target = ScriptedTarget {
            script: vec![0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };

        let requests = vec![
            BatchRequest {
                prompt: vec![1],
                max_new_tokens: 2,
                constraint: None,
            },
            BatchRequest {
                prompt: vec![10; 11],
                max_new_tokens: 3,
                constraint: None,
            },
        ];

        let outputs = engine.generate_batch(&mut draft, &mut target, requests).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].tokens, vec![2, 3]);
        assert_eq!(outputs[1].tokens, vec![11, 12, 13]);

        // Verify isolation: both slots were used
        assert_eq!(draft.bound_slots.len(), 2);
        assert_ne!(draft.bound_slots[0], draft.bound_slots[1]);
    }

    #[test]
    fn test_batch_with_mixed_finishing() {
        let config = EngineConfig {
            draft_window: 1,
            memory: MemoryRuntimeConfig::cpu(1024, 16, 32, 4, 4),
            kv_type: KvCacheType::Fp32,
        };
        let mut engine = VeloEngine::new(config).unwrap();

        // Req 1: generates [2] (stops at max_tokens=1)
        // Req 2: generates [11, 12] (stops at max_tokens=2)
        let mut draft = ScriptedDraft {
            script: vec![0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let mut target = ScriptedTarget {
            script: vec![0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };

        let requests = vec![
            BatchRequest {
                prompt: vec![1],
                max_new_tokens: 1,
                constraint: None,
            },
            BatchRequest {
                prompt: vec![10; 11],
                max_new_tokens: 2,
                constraint: None,
            },
        ];

        let outputs = engine.generate_batch(&mut draft, &mut target, requests).unwrap();

        assert_eq!(outputs[0].tokens, vec![2]);
        assert_eq!(outputs[1].tokens, vec![11, 12]);
    }

    #[test]
    fn test_batch_slot_exhaustion() {
        let config = EngineConfig {
            draft_window: 1,
            memory: MemoryRuntimeConfig::cpu(1024, 16, 32, 1, 1), // Only 1 slot
            kv_type: KvCacheType::Fp32,
        };
        let mut engine = VeloEngine::new(config).unwrap();
        let mut draft = ScriptedDraft {
            script: vec![1, 2],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2],
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        };

        let requests = vec![
            BatchRequest {
                prompt: vec![0],
                max_new_tokens: 1,
                constraint: None,
            },
            BatchRequest {
                prompt: vec![0],
                max_new_tokens: 1,
                constraint: None,
            },
        ];

        let result = engine.generate_batch(&mut draft, &mut target, requests);
        assert!(result.is_err());
    }

    #[test]
    fn engine_error_display() {
        assert!(format!("{}", EngineError::Speculative(SpeculativeError::EmptyDraftWindow)).contains("speculative decoding failed"));
        assert!(format!("{}", EngineError::KvStore(KvStoreError::EmptyBlock)).contains("KV store failed"));
        assert!(format!("{}", EngineError::PagedAttention(PageManagerError::InvalidBlockSize)).contains("paged attention failed"));
    }

    #[test]
    fn engine_accessors_and_stats() {
        let mut engine = VeloEngine::new(EngineConfig {
            draft_window: 1,
            memory: MemoryRuntimeConfig::cpu(16, 16, 32, 1, 32),
            kv_type: KvCacheType::Fp32,
        }).unwrap();
       
        assert_eq!(engine.cache().len(), 0);
        assert_eq!(engine.cache_mut().len(), 0);
       
        let stats = EngineStats::default();
        assert_eq!(stats.cache_hit_tokens, 0);
    }

    #[test]
    fn prefill_empty_prompt() {
        let mut engine = VeloEngine::new(EngineConfig {
            draft_window: 1,
            memory: MemoryRuntimeConfig::cpu(16, 16, 32, 1, 32),
            kv_type: KvCacheType::Fp32,
        }).unwrap();
        let prefill = engine.prefill(&[]).unwrap();
        assert_eq!(prefill.stats.cache_hit_tokens, 0);
        assert!(prefill.inserted_handle.is_none());
    }

    #[test]
    fn generate_batch_with_prompt() {
        let mut engine = VeloEngine::new(EngineConfig {
            draft_window: 1,
            memory: MemoryRuntimeConfig::cpu(16, 16, 32, 1, 32),
            kv_type: KvCacheType::Fp32,
        }).unwrap();
        let mut draft = ScriptedDraft { script: vec![1, 2], bound_prefixes: vec![], bound_slots: vec![] };
        let mut target = ScriptedTarget { script: vec![1, 2], bound_prefixes: vec![], bound_slots: vec![] };
       
        let outputs = engine.generate(&mut draft, &mut target, &[1], 1).unwrap();
        assert_eq!(outputs.tokens.len(), 1);
    }
}
