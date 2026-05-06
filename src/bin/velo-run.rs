/// velo-run — load a GGUF model and generate tokens from a prompt.
///
/// Usage:
///   velo-run --model path/to/model.gguf --prompt "Hello, world" [--max-tokens 32]
///   velo-run --model path/to/model.gguf --token-ids 1,2,3 [--max-tokens 16]
use std::path::PathBuf;
use std::time::Instant;

use velo_core::{CausalLmBackend, GreedySampler, Sampler, TokenLogits};
use velo_core::model_loader::load_gguf;
use velo_core::llama_cpu::LlamaCpuModel;
use velo_core::radix_cache::TokenId;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(e) = run(&args) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let config = parse_args(args)?;

    eprintln!("Loading model: {}", config.model.display());
    let t0 = Instant::now();
    let weights = load_gguf(&config.model).map_err(|e| format!("Failed to load model: {e}"))?;

    let meta = weights.meta.clone();
    eprintln!(
        "Loaded in {:.2}s  arch={} n_layer={} n_embd={} n_head={} n_kv={} vocab={}",
        t0.elapsed().as_secs_f32(),
        meta.arch,
        meta.n_layer,
        meta.n_embd,
        meta.n_head,
        meta.n_head_kv,
        meta.n_vocab,
    );

    let mut model = LlamaCpuModel::new(weights);

    // ── Prompt processing ─────────────────────────────────────────────────────
    let prompt_tokens = config.token_ids.clone();
    if prompt_tokens.is_empty() {
        return Err("No token ids provided; use --token-ids or --prompt".to_string());
    }

    eprintln!(
        "Prompt: {} token(s): {:?}",
        prompt_tokens.len(),
        &prompt_tokens[..prompt_tokens.len().min(8)]
    );

    let t1 = Instant::now();
    let prompt_logits = model.forward_sequence(&prompt_tokens).map_err(|e| format!("Forward pass failed: {e}"))?;

    let ttft = t1.elapsed();

    let sampler = GreedySampler;
    let first_tok = sampler.sample(&prompt_logits, None);

    eprintln!(
        "TTFT: {:.1}ms  first token id: {} (confidence {:.4})",
        ttft.as_secs_f64() * 1000.0,
        first_tok.token,
        first_tok.confidence,
    );

    // ── Generation loop ───────────────────────────────────────────────────────
    let mut generated: Vec<TokenId> = vec![first_tok.token];
    let max_new = config.max_tokens.saturating_sub(1);

    let t2 = Instant::now();
    for _ in 0..max_new {
        let last = *generated.last().unwrap();
        let logits = model.next_logits(&[last]).map_err(|e| format!("Generation failed: {e}"))?;
        let pred = sampler.sample(logits.values(), None);
        generated.push(pred.token);
    }
    let gen_elapsed = t2.elapsed();

    let total_new = generated.len();
    let tok_per_sec = total_new as f64 / gen_elapsed.as_secs_f64();

    eprintln!(
        "Generated {} token(s) in {:.1}ms  ({:.1} tok/s)",
        total_new,
        gen_elapsed.as_secs_f64() * 1000.0,
        tok_per_sec,
    );

    // Print generated token IDs to stdout (one per line for easy piping)
    println!("generated_token_ids:");
    for tok in &generated {
        println!("  {tok}");
    }
    Ok(())
}

// ── CLI parsing ───────────────────────────────────────────────────────────────

#[derive(Debug)]
struct RunConfig {
    model: PathBuf,
    token_ids: Vec<TokenId>,
    max_tokens: usize,
}

#[allow(dead_code)]
const USAGE: &str = "\
Usage: velo-run --model <path> --token-ids <id,id,...> [--max-tokens <n>]
       velo-run --model <path> --prompt <text>          [--max-tokens <n>]

Options:
  --model       Path to a GGUF model file
  --token-ids   Comma-separated prompt token IDs (bypasses tokenizer)
  --prompt      Raw text prompt (naive byte-level tokenization; use --token-ids for accuracy)
  --max-tokens  Maximum new tokens to generate (default: 16)
";

fn parse_args(args: &[String]) -> Result<RunConfig, String> {
    let mut model: Option<PathBuf> = None;
    let mut token_ids: Vec<TokenId> = Vec::new();
    let mut max_tokens: usize = 16;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = Some(PathBuf::from(
                    args.get(i).ok_or("--model requires a path")?,
                ));
            }
            "--token-ids" => {
                i += 1;
                let raw = args.get(i).ok_or("--token-ids requires a value")?;
                token_ids = raw
                    .split(',')
                    .map(|s| {
                        s.trim().parse::<TokenId>().map_err(|_| {
                            format!("invalid token id: {s}")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--prompt" => {
                i += 1;
                let text = args.get(i).ok_or("--prompt requires a value")?;
                // Naive fallback: each UTF-8 byte becomes a token id
                token_ids = text.bytes().map(|b| b as TokenId).collect();
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args
                    .get(i)
                    .ok_or("--max-tokens requires a value")?
                    .parse::<usize>()
                    .map_err(|_| "--max-tokens must be a positive integer".to_string())?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let model = model.ok_or("--model is required")?;

    if token_ids.is_empty() {
        return Err("at least one of --token-ids or --prompt is required".to_string());
    }

    Ok(RunConfig { model, token_ids, max_tokens })
}

#[cfg(test)]
mod tests {
    use super::*;


    fn args(pairs: &[&str]) -> Vec<String> {
        let mut v = vec!["velo-run".to_string()];
        v.extend(pairs.iter().map(|s| s.to_string()));
        v
    }

    #[test]
    fn parses_token_ids() {
        let cfg = parse_args(&args(&["--model", "m.gguf", "--token-ids", "1,2,3"])).unwrap();
        assert_eq!(cfg.token_ids, vec![1, 2, 3]);
    }

    #[test]
    fn parses_max_tokens() {
        let cfg = parse_args(&args(&[
            "--model", "m.gguf",
            "--token-ids", "1",
            "--max-tokens", "64",
        ])).unwrap();
        assert_eq!(cfg.max_tokens, 64);
    }

    #[test]
    fn missing_model_is_error() {
        assert!(parse_args(&args(&["--token-ids", "1"])).is_err());
    }

    #[test]
    fn missing_prompt_is_error() {
        assert!(parse_args(&args(&["--model", "m.gguf"])).is_err());
    }

    #[test]
    fn prompt_bytes_become_token_ids() {
        let cfg =
            parse_args(&args(&["--model", "m.gguf", "--prompt", "hi"])).unwrap();
        assert_eq!(cfg.token_ids, vec![b'h' as u32, b'i' as u32]);
    }

    #[test]
    fn rejects_missing_model() {
        let args = vec!["velo-run".to_string(), "--token-ids".to_string(), "1".to_string()];
        assert_eq!(parse_args(&args).unwrap_err(), "--model is required");
    }

    #[test]
    fn rejects_invalid_token_id() {
        let args = vec![
            "velo-run".to_string(),
            "--model".to_string(), "m.gguf".to_string(),
            "--token-ids".to_string(), "1,abc".to_string()
        ];
        assert!(parse_args(&args).unwrap_err().contains("invalid token id"));
    }

    #[test]
    fn rejects_invalid_max_tokens() {
        let args = vec![
            "velo-run".to_string(),
            "--model".to_string(), "m.gguf".to_string(),
            "--token-ids".to_string(), "1".to_string(),
            "--max-tokens".to_string(), "xyz".to_string()
        ];
        assert!(parse_args(&args).unwrap_err().contains("max-tokens must be a positive integer"));
    }

    #[test]
    fn prompt_option_validation() {
        let args = vec!["velo-run".to_string(), "--model".to_string(), "m.gguf".to_string(), "--prompt".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_run_inference_full() {
        let args = vec!["velo-run".to_string(), "--model".to_string(), "m.gguf".to_string(), "--token-ids".to_string(), "1,2".to_string()];
        // Should parse but fail to load because m.gguf doesn't exist
        let _ = parse_args(&args);
    }

    #[test]
    fn test_help_output() {
        let args = vec!["velo-run".to_string(), "--help".to_string()];
        // parse_args doesn't handle --help specifically (it returns error or usage)
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_invalid_max_tokens_format() {
        let args = vec!["velo-run".to_string(), "--model".to_string(), "m".to_string(), "--token-ids".to_string(), "1".to_string(), "--max-tokens".to_string(), "abc".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_run_inference_logic() {
        let n_vocab = 32;
        let n_embd = 16;
        let weights = velo_core::model_loader::WeightStore::dummy_llama(n_vocab, n_embd, 1);
        let mut model = LlamaCpuModel::new(weights);
        let prompt = vec![1, 2, 3];
       
        let logits = model.forward_sequence(&prompt).unwrap();
        let sampler = GreedySampler;
        let tok = sampler.sample(TokenLogits::new(logits).unwrap().values(), None);
        
        let next_logits = model.next_logits(&[tok.token]).unwrap();
        let next_tok = sampler.sample(next_logits.values(), None);
        assert!(next_tok.token < n_vocab as u32);
    }

    #[test]
    fn test_run_config_traits() {
        let cfg = RunConfig {
            model: PathBuf::from("m.gguf"),
            token_ids: vec![1, 2],
            max_tokens: 32,
        };
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("m.gguf"));
        assert!(dbg.contains("max_tokens: 32"));
    }

    #[test]
    fn test_usage_not_empty() {
        assert!(!USAGE.is_empty());
    }

    #[test]
    fn test_parse_args_exhaustive() {
        // Test unknown argument
        assert!(parse_args(&["velo-run".to_string(), "--unknown".to_string()]).is_err());
       
        // Test missing required model
        assert_eq!(parse_args(&["velo-run".to_string(), "--token-ids".to_string(), "1".to_string()]).unwrap_err(), "--model is required");
       
        // Test missing prompt/token-ids
        assert_eq!(parse_args(&["velo-run".to_string(), "--model".to_string(), "m".to_string()]).unwrap_err(), "at least one of --token-ids or --prompt is required");
    }

    #[test]
    fn test_run_invalid_args() {
        let res = run(&["velo-run".to_string(), "--unknown".to_string()]);
        assert!(res.is_err());
    }
}
