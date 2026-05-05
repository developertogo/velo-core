use std::collections::BTreeMap;

use crate::backend::{CausalLmBackend, TokenLogits};
use crate::radix_cache::{CacheLookup, TokenId};
use crate::speculative::{Result, SpeculativeError};

#[derive(Debug, Clone)]
pub struct MockBackend {
    script: Vec<TokenId>,
    overrides: BTreeMap<usize, TokenId>,
    bound_prefixes: Vec<CacheLookup>,
    bound_slots: Vec<crate::slot_manager::SlotId>,
}

impl MockBackend {
    pub fn new(script: Vec<TokenId>) -> Self {
        Self {
            script,
            overrides: BTreeMap::new(),
            bound_prefixes: Vec::new(),
            bound_slots: Vec::new(),
        }
    }

    pub fn with_override(mut self, position: usize, token: TokenId) -> Self {
        self.overrides.insert(position, token);
        self
    }

    pub fn bound_prefixes(&self) -> &[CacheLookup] {
        &self.bound_prefixes
    }

    fn token_at(&self, position: usize) -> Result<TokenId> {
        if let Some(token) = self.overrides.get(&position) {
            return Ok(*token);
        }

        self.script.get(position).copied().ok_or_else(|| {
            SpeculativeError::Model(format!("mock backend has no token at position {position}"))
        })
    }
}

impl CausalLmBackend for MockBackend {
    fn bind_prefix_cache(&mut self, prefix: &CacheLookup) -> Result<()> {
        self.bound_prefixes.push(prefix.clone());
        Ok(())
    }

    fn bind_slot(&mut self, slot: crate::slot_manager::SlotId) -> Result<()> {
        self.bound_slots.push(slot);
        Ok(())
    }

    fn next_logits(&mut self, context: &[TokenId]) -> Result<TokenLogits> {
        one_hot(self.token_at(context.len())?)
    }

    fn verify_logits(
        &mut self,
        context: &[TokenId],
        drafted: &[TokenId],
    ) -> Result<Vec<TokenLogits>> {
        (0..drafted.len())
            .map(|offset| one_hot(self.token_at(context.len() + offset)?))
            .collect()
    }
}

fn one_hot(token: TokenId) -> Result<TokenLogits> {
    let mut logits = vec![0.0; token as usize + 1];
    logits[token as usize] = 1.0;
    TokenLogits::new(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{GreedyDraftModel, GreedyTargetModel};
    use crate::speculative::{DraftModel, TargetModel, VerifyStep};

    #[test]
    fn emits_scripted_tokens() {
        let backend = MockBackend::new(vec![1, 2, 3]);
        let mut draft = GreedyDraftModel::new(backend);

        let predictions = draft.draft(&[1], 2).unwrap();

        assert_eq!(predictions[0].token, 2);
        assert_eq!(predictions[1].token, 3);
    }

    #[test]
    fn can_override_draft_positions() {
        let backend = MockBackend::new(vec![1, 2, 3]).with_override(1, 9);
        let mut draft = GreedyDraftModel::new(backend);

        let predictions = draft.draft(&[1], 2).unwrap();

        assert_eq!(predictions[0].token, 9);
        assert_eq!(predictions[1].token, 3);
    }

    #[test]
    fn target_verify_reads_unmodified_script() {
        let backend = MockBackend::new(vec![1, 2, 3]);
        let mut target = GreedyTargetModel::new(backend);

        let verified = target.verify(&[1], &[9, 9]).unwrap();

        assert_eq!(
            verified,
            vec![VerifyStep { expected: 2 }, VerifyStep { expected: 3 }]
        );
    }

    #[test]
    fn mock_backend_errors_and_meta() {
        let mut backend = MockBackend::new(vec![1, 2]);
        assert!(backend.token_at(5).is_err());
        
        backend.bind_prefix_cache(&CacheLookup { matched_tokens: 1, handle: None }).unwrap();
        assert_eq!(backend.bound_prefixes().len(), 1);
        
        backend.bind_slot(crate::slot_manager::SlotId(0)).unwrap();
        
        let backend2 = backend.clone();
        assert_eq!(backend2.script, vec![1, 2]);
        assert!(format!("{:?}", backend).contains("MockBackend"));
    }
}
