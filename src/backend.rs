use crate::radix_cache::{CacheLookup, TokenId};
use crate::speculative::{
    DraftModel, NextTokenPrediction, Result, SpeculativeError, TargetModel, VerifyStep,
};

#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogits {
    values: Vec<f32>,
}

pub trait CausalLmBackend {
    fn bind_prefix_cache(&mut self, _prefix: &CacheLookup) -> Result<()> {
        Ok(())
    }

    fn bind_slot(&mut self, _slot: crate::slot_manager::SlotId) -> Result<()> {
        Ok(())
    }

    fn next_logits(&mut self, context: &[TokenId]) -> Result<TokenLogits>;

    fn next_logits_batch(&mut self, contexts: &[&[TokenId]]) -> Result<Vec<TokenLogits>> {
        contexts
            .iter()
            .map(|ctx| self.next_logits(ctx))
            .collect()
    }

    fn verify_logits(&mut self, context: &[TokenId], drafted: &[TokenId])
    -> Result<Vec<TokenLogits>>;

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GreedySampler;

#[derive(Debug, Clone)]
pub struct GreedyDraftModel<B> {
    backend: B,
}

#[derive(Debug, Clone)]
pub struct GreedyTargetModel<B> {
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

impl GreedySampler {
    pub fn sample(&self, logits: &TokenLogits) -> NextTokenPrediction {
        let (token, confidence) = logits
            .values()
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .expect("TokenLogits validates non-empty logits");

        NextTokenPrediction {
            token: token as TokenId,
            confidence,
        }
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

    fn draft(&mut self, context: &[TokenId], max_tokens: usize) -> Result<Vec<NextTokenPrediction>> {
        let sampler = GreedySampler;
        let mut local_context = context.to_vec();
        let mut predictions = Vec::with_capacity(max_tokens);

        for _ in 0..max_tokens {
            let prediction = sampler.sample(&self.backend.next_logits(&local_context)?);
            local_context.push(prediction.token);
            predictions.push(prediction);
        }

        Ok(predictions)
    }

    fn draft_batch(
        &mut self,
        requests: &[(&[TokenId], usize)],
    ) -> Result<Vec<Vec<NextTokenPrediction>>> {
        let sampler = GreedySampler;
        let mut results: Vec<Vec<NextTokenPrediction>> =
            requests.iter().map(|(_, max)| Vec::with_capacity(*max)).collect();
        let mut contexts: Vec<Vec<TokenId>> = requests.iter().map(|(ctx, _)| ctx.to_vec()).collect();
        let max_steps = requests.iter().map(|(_, max)| *max).max().unwrap_or(0);

        for _ in 0..max_steps {
            let active_indices: Vec<usize> = requests
                .iter()
                .enumerate()
                .filter(|(i, (_, max))| results[*i].len() < *max)
                .map(|(i, _)| i)
                .collect();

            if active_indices.is_empty() {
                break;
            }

            let active_contexts: Vec<&[TokenId]> =
                active_indices.iter().map(|&i| contexts[i].as_slice()).collect();

            let batch_logits = self.backend.next_logits_batch(&active_contexts)?;

            for (batch_idx, &req_idx) in active_indices.iter().enumerate() {
                let prediction = sampler.sample(&batch_logits[batch_idx]);
                contexts[req_idx].push(prediction.token);
                results[req_idx].push(prediction);
            }
        }

        Ok(results)
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

    fn verify(&mut self, context: &[TokenId], drafted: &[TokenId]) -> Result<Vec<VerifyStep>> {
        let sampler = GreedySampler;

        Ok(self
            .backend
            .verify_logits(context, drafted)?
            .iter()
            .map(|logits| VerifyStep {
                expected: sampler.sample(logits).token,
            })
            .collect())
    }

    fn verify_batch(
        &mut self,
        requests: &[(&[TokenId], &[TokenId])],
    ) -> Result<Vec<Vec<VerifyStep>>> {
        let sampler = GreedySampler;

        Ok(self
            .backend
            .verify_logits_batch(requests)?
            .into_iter()
            .map(|logits_vec| {
                logits_vec
                    .into_iter()
                    .map(|logits| VerifyStep {
                        expected: sampler.sample(&logits).token,
                    })
                    .collect()
            })
            .collect())
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
            GreedySampler.sample(&logits),
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

        let predictions = draft.draft(&[4], 2).unwrap();

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

        let verified = target.verify(&[2], &[9, 9]).unwrap();

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
        let prediction = GreedySampler.sample(&logits);
        assert_eq!(prediction.token, 0);
        assert_eq!(prediction.confidence, 0.5);
    }

    #[test]
    fn greedy_sampler_picks_last_max() {
        let logits = TokenLogits::new(vec![0.5, 0.5]).unwrap();
        let prediction = GreedySampler.sample(&logits);
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
