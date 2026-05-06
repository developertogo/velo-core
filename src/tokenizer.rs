use std::collections::HashMap;
use crate::gguf::GgufFile;

/// A simple tokenizer that uses the vocabulary from a GGUF file.
#[derive(Clone)]
pub struct Tokenizer {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
}

impl Tokenizer {
    /// Creates a new tokenizer from a GGUF file's metadata.
    pub fn from_gguf(file: &GgufFile) -> Self {
        let mut tokens = Vec::new();
        let mut token_to_id = HashMap::new();

        if let Some(token_array) = file.metadata.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()) {
            for (id, val) in token_array.iter().enumerate() {
                if let Some(s) = val.as_str() {
                    let s = s.to_string();
                    token_to_id.insert(s.clone(), id as u32);
                    tokens.push(s);
                }
            }
        }

        Self { tokens, token_to_id }
    }

    /// Encodes a string into a sequence of token IDs using naive greedy longest-match.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Naive greedy longest-match tokenization for demo purposes.
        // Production engines should use proper BPE/SentencePiece.
        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut matched = false;
            // Try longest matches first
            for i in (1..=remaining.len()).rev() {
                let sub = &remaining[..i];
                if let Some(&id) = self.token_to_id.get(sub) {
                    result.push(id);
                    remaining = &remaining[i..];
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Skip one char if no match (fallback)
                remaining = &remaining[1..];
            }
        }

        result
    }

    /// Decodes a sequence of token IDs back into a string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();
        for &id in tokens {
            if let Some(s) = self.tokens.get(id as usize) {
                result.push_str(s);
            }
        }
        // Clean up common GGUF token artifacts like " " or "<0x0A>"
        result.replace(" ", " ")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufValue};

    fn mock_tokenizer() -> Tokenizer {
        let mut tokens = Vec::new();
        let mut token_to_id = HashMap::new();
       
        let vocab = vec!["<unk>", "<s>", "</s>", " ", "t", "h", "e", "th", "the", "q"];
        for (i, s) in vocab.iter().enumerate() {
            tokens.push(s.to_string());
            token_to_id.insert(s.to_string(), i as u32);
        }
       
        Tokenizer { tokens, token_to_id }
    }

    #[test]
    fn test_encode_greedy() {
        let t = mock_tokenizer();
        // "the" should be matched as one token if possible
        let ids = t.encode("the");
        assert_eq!(ids.len(), 1);
        assert_eq!(t.tokens[ids[0] as usize], "the");

        // "th" should match "th"
        let ids = t.encode("th");
        assert_eq!(ids.len(), 1);
        assert_eq!(t.tokens[ids[0] as usize], "th");
    }

    #[test]
    fn test_decode() {
        let t = mock_tokenizer();
        let ids = vec![8, 9]; // "the", "q"
        assert_eq!(t.decode(&ids), "theq");
    }

    #[test]
    fn test_decode_with_space_artifact() {
        let t = mock_tokenizer();
        let ids = vec![3, 4]; // " ", "t"
        assert_eq!(t.decode(&ids), " t");
    }

    #[test]
    fn test_from_gguf() {
        use std::collections::HashMap;
        let mut metadata = HashMap::new();
        metadata.insert("tokenizer.ggml.tokens".to_string(), GgufValue::Array(vec![
            GgufValue::String("a".into()),
            GgufValue::String("b".into()),
        ]));
       
        let file = GgufFile {
            version: 3,
            metadata,
            tensors: HashMap::new(),
            data_offset: 0,
        };
       
        let t = Tokenizer::from_gguf(&file);
        assert_eq!(t.tokens.len(), 2);
        assert_eq!(t.encode("ab"), vec![0, 1]);
    }
}
