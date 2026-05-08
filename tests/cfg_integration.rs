use velo_core::constraints::{build_factory, LlguidanceMatcher, CfgMatcher};
use velo_core::tokenizer::dummy_tokenizer;
use llguidance::api::TopLevelGrammar;

#[test]
fn test_json_masking_integration() {
    let tokenizer = dummy_tokenizer();
    let factory = build_factory(&tokenizer).expect("Failed to build factory");
    
    // Simple JSON schema: an object with "name" string and "age" number.
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "number" }
        },
        "required": ["name", "age"]
    });
    
    let grammar = TopLevelGrammar::from_json_schema(schema);
    let mut matcher = LlguidanceMatcher::new(&factory, grammar, tokenizer.vocab_size())
        .expect("Failed to create matcher");
        
    // First token MUST be '{'
    let mask = matcher.next_mask();
    let brace_id = tokenizer.encode("{")[0] as usize;
    assert!(mask.get(brace_id).unwrap(), "First token must allow '{{'");
    
    // Ensure it DOES NOT allow '}' at the very start
    let close_brace_id = tokenizer.encode("}")[0] as usize;
    assert!(!mask.get(close_brace_id).unwrap_or(false), "First token must NOT allow '}}'");

    // Advance with '{'
    matcher.advance(brace_id as u32);
    let mask = matcher.next_mask();
    let quote_id = tokenizer.encode("\"")[0] as usize;
    assert!(mask.get(quote_id).unwrap(), "After '{{', must allow '\"'");
    
    println!("CFG Integration Test Passed!");
}
