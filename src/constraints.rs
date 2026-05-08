use std::sync::Arc;
use toktrie::TokTrie;
use llguidance::api::TopLevelGrammar;
use llguidance::ParserFactory;
use anyhow::Result;
use crate::radix_cache::TokenId;
use crate::sampling::LogitMask;

/// Represents a constraint on generation (Regex, JSON Schema, or Lark Grammar).
#[derive(Debug, Clone)]
pub enum Constraint {
    Regex(String),
    JsonSchema(serde_json::Value),
    Lark(String),
}

/// Orchestrates constrained generation by providing logit masks at each step.
pub trait CfgMatcher: std::fmt::Debug + Send + Sync {
    /// Returns the mask of allowed tokens for the current state.
    fn next_mask(&mut self) -> LogitMask;
    
    /// Advances the matcher's state by consuming the selected token.
    fn advance(&mut self, token: TokenId);

    /// Clones the matcher to a new Box.
    fn clone_box(&self) -> Box<dyn CfgMatcher>;
}

/// A matcher based on llguidance.
pub struct LlguidanceMatcher {
    parser: llguidance::Matcher,
    vocab_size: usize,
}

impl std::fmt::Debug for LlguidanceMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlguidanceMatcher")
            .field("vocab_size", &self.vocab_size)
            .finish()
    }
}

impl LlguidanceMatcher {
    pub fn new(factory: &ParserFactory, grammar: TopLevelGrammar, vocab_size: usize) -> Result<Self> {
        let parser = factory.create_parser(grammar)?;
        Ok(Self {
            parser: llguidance::Matcher::new(Ok(parser)),
            vocab_size,
        })
    }
}

impl CfgMatcher for LlguidanceMatcher {
    fn next_mask(&mut self) -> LogitMask {
        let mut mask = LogitMask::from_elem(false, self.vocab_size);
        match self.parser.compute_mask_or_eos() {
            Ok(llg_mask) => {
                llg_mask.iter_set_entries(|idx| {
                    if idx < self.vocab_size {
                        mask.set(idx, true);
                    }
                });
            }
            Err(_) => {
                mask.set_all(true);
            }
        }
        mask
    }

    fn advance(&mut self, token: TokenId) {
        let _ = self.parser.consume_token(token);
    }

    fn clone_box(&self) -> Box<dyn CfgMatcher> {
        Box::new(LlguidanceMatcher {
            parser: self.parser.clone(),
            vocab_size: self.vocab_size,
        })
    }
}

/// Helper to build the llguidance ParserFactory from our Tokenizer.
pub fn build_factory(tokenizer: &crate::tokenizer::Tokenizer) -> Result<Arc<ParserFactory>> {
    let vocab_size = tokenizer.vocab_size();
    let mut token_bytes = Vec::with_capacity(vocab_size);
    
    for i in 0..vocab_size {
        let bytes = tokenizer.id_to_token(i as u32).as_bytes().to_vec();
        token_bytes.push(bytes);
    }
    
    // TokRxInfo needs vocab_size, tok_eos, tok_bos, tok_unk, tok_end_of_turn, tok_pad.
    let info = toktrie::TokRxInfo {
        vocab_size: vocab_size as u32,
        tok_eos: 2, // Llama default
        tok_bos: Some(1),
        tok_unk: Some(0),
        tok_end_of_turn: None,
        tok_pad: None,
    };
    
    let tok_trie = TokTrie::from(&info, &token_bytes);
    let base_env = Arc::new(toktrie::ApproximateTokEnv::new(tok_trie.clone()));
    let env: Arc<dyn toktrie::TokenizerEnv + Sync> = Arc::new(toktrie::TokEnvWithTrie::new(base_env, tok_trie));
    
    // Create ParserFactory with default limits
    let factory = ParserFactory::new(&env, Default::default(), &[])?;
    Ok(Arc::new(factory))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_enum() {
        let regex = Constraint::Regex("a*".to_string());
        assert!(format!("{:?}", regex).contains("Regex"));
        
        let json = Constraint::JsonSchema(serde_json::json!({"type": "string"}));
        assert!(format!("{:?}", json).contains("JsonSchema"));
        
        let lark = Constraint::Lark("start: /a/".to_string());
        assert!(format!("{:?}", lark).contains("Lark"));
    }
}
