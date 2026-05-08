use crate::radix_cache::{CacheLookup, TokenId};
use crate::speculative::{
    DraftModel, NextTokenPrediction, Result, SpeculativeError, TargetModel, VerifyStep,
};
use crate::sampling::{Sampler, GreedySampler};

/// A collection of raw logit values for the vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogits {
    values: Vec<f32>,
}

/// Core trait for causal language model backends.
/// 
/// This trait provides the interface for both prefill and decode phases,
/// as well as verification steps for speculative decoding.
pub trait CausalLmBackend {
    /// Binds a prefix cache lookup result to the current session.
    /// This allows the backend to reuse KV-cache pages for prompt prefixes.
    fn bind_prefix_cache(&mut self, _prefix: &CacheLookup) -> Result<()> {
        Ok(())
    }

    /// Binds the backend to a specific inference slot.
    fn bind_slot(&mut self, _slot: crate::slot_manager::SlotId) -> Result<()> {
        Ok(())
    }

    /// Switches the active model weights in the backend.
    fn switch_model(&mut self, _name: &str, _pool: &crate::model_pool::ModelPool) -> Result<()> {
        Ok(())
    }

    /// Performs a single forward pass to get the logits for the next token.
    fn next_logits(&mut self, context: &[TokenId]) -> Result<TokenLogits>;

    /// Samples the next token from the model using the provided sampler.
    fn sample_next(&mut self, context: &[TokenId], sampler: &dyn Sampler) -> Result<NextTokenPrediction> {
        let logits = self.next_logits(context)?;
        Ok(sampler.sample(logits.values(), None))
    }

    /// Performs a batched forward pass for multiple sequences.
    fn next_logits_batch(&mut self, contexts: &[&[TokenId]]) -> Result<Vec<TokenLogits>> {
        contexts
            .iter()
            .map(|ctx| self.next_logits(ctx))
            .collect()
    }

    /// Performs a verification pass for a drafted sequence.
    /// Returns a list of logits, one for each token in the context + drafted sequence.
    fn verify_logits(&mut self, context: &[TokenId], drafted: &[TokenId])
    -> Result<Vec<TokenLogits>>;

    /// Performs a batched verification pass for multiple drafted sequences.
    fn verify_logits_batch(
        &mut self,
        requests: &[(&[TokenId], &[TokenId])],
    ) -> Result<Vec<Vec<TokenLogits>>> {
        requests
            .iter()
            .map(|(ctx, drafted)| self.verify_logits(ctx, drafted))
            .collect()
    }
}

impl<B: CausalLmBackend + ?Sized> CausalLmBackend for &mut B {
    fn next_logits(&mut self, context: &[TokenId]) -> Result<TokenLogits> {
        (**self).next_logits(context)
    }

    fn verify_logits(&mut self, context: &[TokenId], drafted: &[TokenId]) -> Result<Vec<TokenLogits>> {
        (**self).verify_logits(context, drafted)
    }
}


#[derive(Debug, Clone)]
pub struct GreedyDraftModel<B> {
    backend: B,
}

#[derive(Debug, Clone)]
pub struct GreedyTargetModel<B> {
    backend: B,
}

#[derive(Debug, Clone)]
pub struct TreeDraftModel<B> {
    backend: B,
}

impl TokenLogits {
    pub fn new(values: Vec<f32>) -> Result<Self> {
        if values.is_empty() {
            return Err(SpeculativeError::Model(
                "logits must contain at least one token".to_string(),
            ));
        }

        Ok(Self { values })
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}


impl<B> GreedyDraftModel<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B> GreedyTargetModel<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B> DraftModel for GreedyDraftModel<B>
where
    B: CausalLmBackend,
{
    fn bind_prefix_cache(&mut self, prefix: &CacheLookup) -> Result<()> {
        self.backend.bind_prefix_cache(prefix)
    }

    fn bind_slot(&mut self, slot: crate::slot_manager::SlotId) -> Result<()> {
        self.backend.bind_slot(slot)
    }

    fn switch_model(&mut self, name: &str, pool: &crate::model_pool::ModelPool) -> Result<()> {
        self.backend.switch_model(name, pool)
    }

    fn draft(
        &mut self,
        context: &[TokenId],
        max_tokens: usize,
        mut matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>,
    ) -> Result<Vec<NextTokenPrediction>> {
        let sampler = GreedySampler;
        let mut local_context = context.to_vec();
        let mut predictions = Vec::with_capacity(max_tokens);

        for _ in 0..max_tokens {
            let mask = matcher.as_mut().map(|m| m.next_mask());
            let prediction = sampler.sample(
                self.backend.next_logits(&local_context)?.values(),
                mask.as_ref(),
            );
            local_context.push(prediction.token);
            predictions.push(prediction);
            if let Some(m) = matcher.as_mut() {
                m.advance(prediction.token);
            }
        }

        Ok(predictions)
    }

    fn draft_batch(
        &mut self,
        requests: &mut [(&[TokenId], usize, Option<&mut (dyn crate::constraints::CfgMatcher + '_)>)],
    ) -> Result<Vec<Vec<NextTokenPrediction>>> {
        let sampler = GreedySampler;
        let mut results: Vec<Vec<NextTokenPrediction>> =
            requests.iter().map(|(_, max, _)| Vec::with_capacity(*max)).collect();
        let mut contexts: Vec<Vec<TokenId>> = requests.iter().map(|(ctx, _, _)| ctx.to_vec()).collect();
        let max_steps = requests.iter().map(|(_, max, _)| *max).max().unwrap_or(0);

        for _ in 0..max_steps {
            let active_indices: Vec<usize> = requests
                .iter()
                .enumerate()
                .filter(|(i, (_, max, _))| results[*i].len() < *max)
                .map(|(i, _)| i)
                .collect();

            if active_indices.is_empty() {
                break;
            }

            let active_contexts: Vec<&[TokenId]> =
                active_indices.iter().map(|&i| contexts[i].as_slice()).collect();

            let batch_logits = self.backend.next_logits_batch(&active_contexts)?;

            for (batch_idx, &req_idx) in active_indices.iter().enumerate() {
                let mask = requests[req_idx].2.as_mut().map(|m| m.next_mask());
                let mask_ref = mask.as_ref();
                let prediction = sampler.sample(batch_logits[batch_idx].values(), mask_ref);
                if let Some(m) = mask_ref {
                    if !m.get(prediction.token as usize).unwrap_or(false) {
                        eprintln!("Sampler returned token {} which is NOT allowed by mask!", prediction.token);
                    }
                }
                contexts[req_idx].push(prediction.token);
                results[req_idx].push(prediction);
                if let Some(m) = requests[req_idx].2.as_mut() {
                    m.advance(prediction.token);
                }
            }
        }

        Ok(results)
    }
}

impl<B> DraftModel for TreeDraftModel<B>
where
    B: CausalLmBackend,
{
    fn bind_prefix_cache(&mut self, prefix: &CacheLookup) -> Result<()> {
        self.backend.bind_prefix_cache(prefix)
    }

    fn bind_slot(&mut self, slot: crate::slot_manager::SlotId) -> Result<()> {
        self.backend.bind_slot(slot)
    }

    fn switch_model(&mut self, name: &str, pool: &crate::model_pool::ModelPool) -> Result<()> {
        self.backend.switch_model(name, pool)
    }

    fn draft(
        &mut self,
        context: &[TokenId],
        max_tokens: usize,
        matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>,
    ) -> Result<Vec<NextTokenPrediction>> {
        let mut greedy = GreedyDraftModel::new(&mut self.backend);
        greedy.draft(context, max_tokens, matcher)
    }

    fn draft_tree(
        &mut self,
        context: &[TokenId],
        max_tokens: usize,
        width: usize,
        mut matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>,
    ) -> Result<crate::speculative::SpeculativeTree> {
        use crate::sampling::TopKSampler;
        let sampler = TopKSampler { k: width };
        
        let mask = matcher.as_mut().map(|m| m.next_mask());
        let top_k = sampler.sample_top_k(self.backend.next_logits(context)?.values(), mask.as_ref());
        
        if top_k.is_empty() {
            return Err(SpeculativeError::Model("Draft model returned no tokens".into()));
        }

        let mut t = crate::speculative::SpeculativeTree::new(top_k[0].token);
        
        // Add other top-K as branches from the start
        for pred in top_k.into_iter().skip(1) {
            t.add_child(0, pred.token);
        }
        
        // Greedily expand the primary branch
        let mut curr = 0;
        for _ in 1..max_tokens {
            let path = t.get_path(curr);
            let mut full_context = context.to_vec();
            full_context.extend_from_slice(&path);
            
            // Note: CFG matcher advancement is simplified here
            let next_logits = self.backend.next_logits(&full_context)?;
            let next_token = sampler.sample(next_logits.values(), None);
            
            curr = t.add_child(curr, next_token.token);
        }

        Ok(t)
    }
}

impl<B> TargetModel for GreedyTargetModel<B>
where
    B: CausalLmBackend,
{
    fn bind_prefix_cache(&mut self, prefix: &CacheLookup) -> Result<()> {
        self.backend.bind_prefix_cache(prefix)
    }

    fn bind_slot(&mut self, slot: crate::slot_manager::SlotId) -> Result<()> {
        self.backend.bind_slot(slot)
    }

    fn switch_model(&mut self, name: &str, pool: &crate::model_pool::ModelPool) -> Result<()> {
        self.backend.switch_model(name, pool)
    }

    fn verify(
        &mut self,
        context: &[TokenId],
        drafted: &[TokenId],
        mut matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>,
    ) -> Result<Vec<VerifyStep>> {
        let sampler = GreedySampler;
        let logits_vec = self.backend.verify_logits(context, drafted)?;
        let mut steps = Vec::with_capacity(drafted.len());

        for logits in logits_vec {
            let mask = matcher.as_mut().map(|m| m.next_mask());
            let prediction = sampler.sample(logits.values(), mask.as_ref());
            steps.push(VerifyStep {
                expected: prediction.token,
            });
            if let Some(m) = matcher.as_mut() {
                m.advance(prediction.token);
            }
        }

        Ok(steps)
    }

    fn verify_batch(
        &mut self,
        requests: &mut [(&[TokenId], &[TokenId], Option<&mut (dyn crate::constraints::CfgMatcher + '_)>)],
    ) -> Result<Vec<Vec<VerifyStep>>> {
        let sampler = GreedySampler;
        let verify_reqs: Vec<(&[TokenId], &[TokenId])> = requests.iter().map(|(ctx, drafted, _)| (*ctx, *drafted)).collect();
        
        let batch_logits = self.backend.verify_logits_batch(&verify_reqs)?;
        let mut results = Vec::with_capacity(requests.len());

        for (i, logits_vec) in batch_logits.into_iter().enumerate() {
            let mut steps = Vec::with_capacity(logits_vec.len());
            for logits in logits_vec {
                let mask = requests[i].2.as_mut().map(|m| m.next_mask());
                let mask_ref = mask.as_ref();
                let prediction = sampler.sample(logits.values(), mask_ref);
                if let Some(m) = mask_ref {
                    if !m.get(prediction.token as usize).unwrap_or(false) {
                        eprintln!("TARGET Sampler returned token {} which is NOT allowed by mask!", prediction.token);
                    }
                }
                steps.push(VerifyStep {
                    expected: prediction.token,
                });
                if let Some(m) = requests[i].2.as_mut() {
                    m.advance(prediction.token);
                }
            }
            results.push(steps);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct ScriptedBackend {
        script: Vec<TokenId>,
        bound_prefixes: Vec<CacheLookup>,
    }

    impl CausalLmBackend for ScriptedBackend {
        fn bind_prefix_cache(&mut self, prefix: &CacheLookup) -> Result<()> {
            self.bound_prefixes.push(prefix.clone());
            Ok(())
        }

        fn next_logits(&mut self, context: &[TokenId]) -> Result<TokenLogits> {
            one_hot(self.script[context.len()])
        }

        fn verify_logits(
            &mut self,
            context: &[TokenId],
            drafted: &[TokenId],
        ) -> Result<Vec<TokenLogits>> {
            self.script[context.len()..]
                .iter()
                .take(drafted.len())
                .map(|token| one_hot(*token))
                .collect()
        }
    }

    fn one_hot(token: TokenId) -> Result<TokenLogits> {
        let mut logits = vec![0.0; token as usize + 1];
        logits[token as usize] = 1.0;
        TokenLogits::new(logits)
    }

    #[test]
    fn greedy_sampler_selects_highest_logit() {
        let logits = TokenLogits::new(vec![0.1, 0.8, 0.2]).unwrap();

        assert_eq!(
            GreedySampler.sample(logits.values(), None),
            NextTokenPrediction {
                token: 1,
                confidence: 0.8,
            }
        );
    }

    #[test]
    fn greedy_draft_model_autoregressively_extends_context() {
        let backend = ScriptedBackend {
            script: vec![4, 5, 6],
            bound_prefixes: Vec::new(),
        };
        let mut draft = GreedyDraftModel::new(backend);

        let predictions = draft.draft(&[4], 2, None).unwrap();

        assert_eq!(
            predictions,
            vec![
                NextTokenPrediction {
                    token: 5,
                    confidence: 1.0,
                },
                NextTokenPrediction {
                    token: 6,
                    confidence: 1.0,
                },
            ]
        );
    }

    #[test]
    fn greedy_target_model_converts_verify_logits_to_expected_tokens() {
        let backend = ScriptedBackend {
            script: vec![2, 3, 4],
            bound_prefixes: Vec::new(),
        };
        let mut target = GreedyTargetModel::new(backend);

        let verified = target.verify(&[2], &[9, 9], None).unwrap();

        assert_eq!(
            verified,
            vec![VerifyStep { expected: 3 }, VerifyStep { expected: 4 }]
        );
    }

    #[test]
    fn adapters_forward_prefix_cache_bindings() {
        let prefix = CacheLookup {
            matched_tokens: 2,
            handle: None,
        };
        let backend = ScriptedBackend {
            script: vec![1, 2],
            bound_prefixes: Vec::new(),
        };
        let mut draft = GreedyDraftModel::new(backend);

        draft.bind_prefix_cache(&prefix).unwrap();

        assert_eq!(draft.backend().bound_prefixes, vec![prefix]);
    }

    #[test]
    fn token_logits_rejects_empty_values() {
        assert!(TokenLogits::new(vec![]).is_err());
    }

    #[test]
    fn greedy_sampler_handles_single_logit() {
        let logits = TokenLogits::new(vec![0.5]).unwrap();
        let prediction = GreedySampler.sample(logits.values(), None);
        assert_eq!(prediction.token, 0);
        assert_eq!(prediction.confidence, 0.5);
    }

    #[test]
    fn greedy_sampler_picks_last_max() {
        let logits = TokenLogits::new(vec![0.5, 0.5]).unwrap();
        let prediction = GreedySampler.sample(logits.values(), None);
        assert_eq!(prediction.token, 1);
    }

    #[test]
    fn next_logits_batch_defaults_to_sequential() {
        struct SequentialBackend;
        impl CausalLmBackend for SequentialBackend {
            fn next_logits(&mut self, ctx: &[TokenId]) -> Result<TokenLogits> {
                TokenLogits::new(vec![ctx.len() as f32])
            }
            fn verify_logits(&mut self, _: &[TokenId], _: &[TokenId]) -> Result<Vec<TokenLogits>> {
                Ok(vec![])
            }
        }
        let mut backend = SequentialBackend;
        let batch = backend.next_logits_batch(&[&[1], &[1, 2]]).unwrap();
        assert_eq!(batch[0].values(), &[1.0]);
        assert_eq!(batch[1].values(), &[2.0]);
    }

    #[test]
    fn verify_logits_batch_defaults_to_sequential() {
        struct SequentialBackend;
        impl CausalLmBackend for SequentialBackend {
            fn next_logits(&mut self, _: &[TokenId]) -> Result<TokenLogits> {
                TokenLogits::new(vec![0.0])
            }
            fn verify_logits(&mut self, ctx: &[TokenId], drafted: &[TokenId]) -> Result<Vec<TokenLogits>> {
                Ok(vec![TokenLogits::new(vec![(ctx.len() + drafted.len()) as f32]).unwrap()])
            }
        }
        let mut backend = SequentialBackend;
        let batch = backend.verify_logits_batch(&[(&[1], &[2]), (&[1, 2], &[3])]).unwrap();
        assert_eq!(batch[0][0].values(), &[2.0]);
        assert_eq!(batch[1][0].values(), &[3.0]);
    }
}
