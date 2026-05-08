use velo_core::engine::{VeloEngine, EngineConfig, BatchRequest};
use velo_core::constraints::{build_factory, Constraint};
use velo_core::tokenizer::dummy_tokenizer;
use velo_core::backend::{GreedyDraftModel, GreedyTargetModel, CausalLmBackend, TokenLogits};
use velo_core::radix_cache::TokenId;
use velo_core::speculative::Result as SpecResult;
use velo_core::paged_attention::KvCacheType;
use velo_core::runtime::MemoryRuntimeConfig;

#[derive(Debug, Clone)]
struct JSONPreferenceBackend {
    tokenizer: velo_core::tokenizer::Tokenizer,
}

impl CausalLmBackend for JSONPreferenceBackend {
    fn next_logits(&mut self, context: &[TokenId]) -> SpecResult<TokenLogits> {
        let text = self.tokenizer.decode(context);
        let mut logits = vec![-10.0; self.tokenizer.vocab_size()];
        
        // Strategy: Model "prefers" invalid JSON but the mask should stop it.
        // If we just had '{ "name": ', the model might want to output 'Alice' without quotes.
        if text.ends_with(": ") {
            let a_id = self.tokenizer.encode("a")[0] as usize;
            let quote_id = self.tokenizer.encode("\"")[0] as usize;
            println!("At ': ', a_id={}, quote_id={}", a_id, quote_id);
            logits[a_id] = 10.0;
            logits[quote_id] = 5.0;
        } else if text.ends_with(": \"") {
            // Model outputs 'alice'
            let alice_id = self.tokenizer.encode("alice")[0] as usize;
            logits[alice_id] = 10.0;
        } else if text.ends_with("alice") {
            // Model wants ' ' (invalid, must close quote)
            let space_id = self.tokenizer.encode(" ")[0] as usize;
            logits[space_id] = 10.0;
            
            // Mask should force '"'
            let quote_id = self.tokenizer.encode("\"")[0] as usize;
            logits[quote_id] = 5.0;
        } else {
            // Default: just output something
            logits[0] = 0.0;
        }
        
        Ok(TokenLogits::new(logits).unwrap())
    }

    fn verify_logits(&mut self, context: &[TokenId], drafted: &[TokenId]) -> SpecResult<Vec<TokenLogits>> {
        let mut results = Vec::new();
        let mut current_ctx = context.to_vec();
        for &t in drafted {
            results.push(self.next_logits(&current_ctx)?);
            current_ctx.push(t);
        }
        Ok(results)
    }
}

#[test]
fn test_json_e2e_constrained_generation() {
    let tokenizer = dummy_tokenizer();
    let factory = build_factory(&tokenizer).expect("Failed to build factory");
    
    let mut engine = VeloEngine::new(EngineConfig {
        draft_window: 4,
        memory: MemoryRuntimeConfig::cpu(128, 16, 32, 4, 4),
        kv_type: KvCacheType::Fp32,
    }).unwrap();
    engine.parser_factory = Some(factory);

    let backend = JSONPreferenceBackend { tokenizer: tokenizer.clone() };
    let mut draft = GreedyDraftModel::new(backend.clone());
    let mut target = GreedyTargetModel::new(backend);

    let prompt_text = "{ \"name\": ";
    let prompt = tokenizer.encode(prompt_text);
    println!("Prompt text: '{}'", prompt_text);
    println!("Prompt tokens: {:?}", prompt);
    for &t in &prompt {
        println!("  token {}: '{}'", t, tokenizer.id_to_token(t));
    }
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        }
    });

    let request = BatchRequest {
        prompt: prompt.clone(),
        max_new_tokens: 10,
        constraint: Some(Constraint::JsonSchema(schema)),
    };

    let outputs = engine.generate_batch(&mut draft, &mut target, vec![request]).unwrap();
    let result_text = tokenizer.decode(&outputs[0].tokens);
    
    // It should have forced the quote at the start
    assert!(result_text.starts_with("\""), "Generated text should start with a quote: {}", result_text);
    println!("CFG E2E Test Passed! Generated: {}", result_text);
}
