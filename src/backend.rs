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

    fn verify_logits(&mut self, context: &[TokenId], drafted: &[TokenId])
    -> Result<Vec<TokenLogits>>;
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
}
