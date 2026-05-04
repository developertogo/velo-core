use crate::radix_cache::{CacheLookup, TokenId};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NextTokenPrediction {
    pub token: TokenId,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifyStep {
    pub expected: TokenId,
}

pub trait DraftModel {
    fn bind_prefix_cache(&mut self, _prefix: &CacheLookup) -> Result<()> {
        Ok(())
    }

    fn bind_slot(&mut self, _slot: crate::slot_manager::SlotId) -> Result<()> {
        Ok(())
    }

    fn draft(&mut self, context: &[TokenId], max_tokens: usize) -> Result<Vec<NextTokenPrediction>>;
}

pub trait TargetModel {
    fn bind_prefix_cache(&mut self, _prefix: &CacheLookup) -> Result<()> {
        Ok(())
    }

    fn bind_slot(&mut self, _slot: crate::slot_manager::SlotId) -> Result<()> {
        Ok(())
    }

    fn verify(&mut self, context: &[TokenId], drafted: &[TokenId]) -> Result<Vec<VerifyStep>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeDecoder {
    draft_window: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeSession {
    draft_window: usize,
    prompt: Vec<TokenId>,
    context: Vec<TokenId>,
    stats: SpeculativeStats,
    rejected_token: Option<TokenId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeOutput {
    pub tokens: Vec<TokenId>,
    pub stats: SpeculativeStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    pub draft_calls: usize,
    pub target_calls: usize,
    pub drafted_tokens: usize,
    pub accepted_tokens: usize,
    pub rejected_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculativeError {
    EmptyDraftWindow,
    DraftReturnedTooMany {
        requested: usize,
        returned: usize,
    },
    TargetReturnedWrongLength {
        drafted: usize,
        returned: usize,
    },
    Model(String),
}

pub type Result<T> = std::result::Result<T, SpeculativeError>;

impl SpeculativeDecoder {
    pub fn new(draft_window: usize) -> Result<Self> {
        if draft_window == 0 {
            return Err(SpeculativeError::EmptyDraftWindow);
        }

        Ok(Self { draft_window })
    }

    pub fn generate<D, T>(
        &self,
        draft_model: &mut D,
        target_model: &mut T,
        prompt: &[TokenId],
        max_new_tokens: usize,
    ) -> Result<SpeculativeOutput>
    where
        D: DraftModel,
        T: TargetModel,
    {
        let mut session = self.begin(prompt)?;
        let mut generated = Vec::with_capacity(max_new_tokens);

        while generated.len() < max_new_tokens {
            let remaining = max_new_tokens - generated.len();
            let drafted = session.draft(draft_model, target_model, remaining)?;

            if drafted.is_empty() && !session.has_pending_rejection() {
                break;
            }

            generated.extend_from_slice(&drafted);
            if !session.has_pending_rejection() {
                continue;
            }

            if let Some(token) = session.take_rejected_token() {
                generated.push(token);
            }
        }

        Ok(SpeculativeOutput {
            tokens: generated,
            stats: session.stats().clone(),
        })
    }

    pub fn begin(&self, prompt: &[TokenId]) -> Result<SpeculativeSession> {
        Ok(SpeculativeSession {
            draft_window: self.draft_window,
            prompt: prompt.to_vec(),
            context: prompt.to_vec(),
            stats: SpeculativeStats::default(),
            rejected_token: None,
        })
    }
}

impl SpeculativeSession {
    pub fn prompt(&self) -> &[TokenId] {
        &self.prompt
    }

    pub fn context(&self) -> &[TokenId] {
        &self.context
    }

    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    pub fn has_pending_rejection(&self) -> bool {
        self.rejected_token.is_some()
    }

    pub fn take_rejected_token(&mut self) -> Option<TokenId> {
        self.rejected_token.take()
    }

    pub fn draft<D, T>(
        &mut self,
        draft_model: &mut D,
        target_model: &mut T,
        max_new_tokens: usize,
    ) -> Result<Vec<TokenId>>
    where
        D: DraftModel,
        T: TargetModel,
    {
        let requested = max_new_tokens.min(self.draft_window);
        let predictions = draft_model.draft(&self.context, requested)?;
        self.stats.draft_calls += 1;

        if predictions.len() > requested {
            return Err(SpeculativeError::DraftReturnedTooMany {
                requested,
                returned: predictions.len(),
            });
        }

        if predictions.is_empty() {
            return Ok(Vec::new());
        }

        let drafted = predictions
            .iter()
            .map(|prediction| prediction.token)
            .collect::<Vec<_>>();
        self.stats.drafted_tokens += drafted.len();

        let verified = target_model.verify(&self.context, &drafted)?;
        self.stats.target_calls += 1;

        if verified.len() != drafted.len() {
            return Err(SpeculativeError::TargetReturnedWrongLength {
                drafted: drafted.len(),
                returned: verified.len(),
            });
        }

        let mut accepted_this_round = 0;
        let mut rejected = None;

        for (index, (draft_token, target_step)) in drafted.iter().zip(&verified).enumerate() {
            if *draft_token == target_step.expected {
                accepted_this_round += 1;
                continue;
            }

            rejected = Some((index, target_step.expected));
            break;
        }

        let mut accepted = Vec::with_capacity(accepted_this_round);
        if accepted_this_round > 0 {
            accepted.extend_from_slice(&drafted[..accepted_this_round]);
            self.context.extend_from_slice(&accepted);
            self.stats.accepted_tokens += accepted_this_round;
        }

        if let Some((rejected_index, target_token)) = rejected {
            self.context.push(target_token);
            self.rejected_token = Some(target_token);
            self.stats.rejected_tokens += drafted.len() - rejected_index;
        } else {
            self.rejected_token = None;
        }

        Ok(accepted)
    }
}

impl std::fmt::Display for SpeculativeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDraftWindow => write!(formatter, "draft window must be greater than zero"),
            Self::DraftReturnedTooMany {
                requested,
                returned,
            } => write!(
                formatter,
                "draft model returned {returned} tokens after {requested} were requested"
            ),
            Self::TargetReturnedWrongLength { drafted, returned } => write!(
                formatter,
                "target model returned {returned} verification steps for {drafted} drafted tokens"
            ),
            Self::Model(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for SpeculativeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct ScriptedDraft {
        script: Vec<TokenId>,
    }

    impl DraftModel for ScriptedDraft {
        fn draft(
            &mut self,
            context: &[TokenId],
            max_tokens: usize,
        ) -> Result<Vec<NextTokenPrediction>> {
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
    }

    impl TargetModel for ScriptedTarget {
        fn verify(&mut self, context: &[TokenId], drafted: &[TokenId]) -> Result<Vec<VerifyStep>> {
            Ok(self.script[context.len()..]
                .iter()
                .take(drafted.len())
                .map(|token| VerifyStep { expected: *token })
                .collect())
        }
    }

    #[test]
    fn accepts_full_draft_windows() {
        let decoder = SpeculativeDecoder::new(3).unwrap();
        let prompt = [1, 2];
        let mut draft = ScriptedDraft {
            script: vec![1, 2, 3, 4, 5, 6],
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3, 4, 5, 6],
        };

        let output = decoder
            .generate(&mut draft, &mut target, &prompt, 4)
            .unwrap();

        assert_eq!(output.tokens, vec![3, 4, 5, 6]);
        assert_eq!(
            output.stats,
            SpeculativeStats {
                draft_calls: 2,
                target_calls: 2,
                drafted_tokens: 4,
                accepted_tokens: 4,
                rejected_tokens: 0,
            }
        );
    }

    #[test]
    fn falls_back_to_target_token_on_first_rejection() {
        let decoder = SpeculativeDecoder::new(4).unwrap();
        let prompt = [1, 2];
        let mut draft = ScriptedDraft {
            script: vec![1, 2, 9, 4, 5],
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3, 4, 5],
        };

        let output = decoder
            .generate(&mut draft, &mut target, &prompt, 3)
            .unwrap();

        assert_eq!(output.tokens, vec![3, 4, 5]);
        assert_eq!(output.stats.accepted_tokens, 2);
        assert_eq!(output.stats.rejected_tokens, 3);
    }

    #[test]
    fn stops_at_max_new_tokens() {
        let decoder = SpeculativeDecoder::new(8).unwrap();
        let prompt = [1];
        let mut draft = ScriptedDraft {
            script: vec![1, 2, 3, 4, 5],
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3, 4, 5],
        };

        let output = decoder
            .generate(&mut draft, &mut target, &prompt, 2)
            .unwrap();

        assert_eq!(output.tokens, vec![2, 3]);
        assert_eq!(output.stats.drafted_tokens, 2);
    }

    #[test]
    fn rejects_zero_draft_window() {
        assert_eq!(
            SpeculativeDecoder::new(0),
            Err(SpeculativeError::EmptyDraftWindow)
        );
    }

    #[test]
    fn session_exposes_prompt_and_context() {
        let decoder = SpeculativeDecoder::new(2).unwrap();
        let session = decoder.begin(&[7, 8]).unwrap();

        assert_eq!(session.prompt(), &[7, 8]);
        assert_eq!(session.context(), &[7, 8]);
    }
}
