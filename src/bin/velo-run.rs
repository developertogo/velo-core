/// velo-run — load a GGUF model and generate tokens from a prompt.
///
/// Usage:
///   velo-run --model path/to/model.gguf --prompt "Hello, world" [--max-tokens 32]
///   velo-run --model path/to/model.gguf --token-ids 1,2,3 [--max-tokens 16]
use std::path::PathBuf;
use std::time::Instant;

use velo_core::backend::{CausalLmBackend, GreedySampler, TokenLogits};
use velo_core::model_loader::load_gguf;
use velo_core::llama_cpu::LlamaCpuModel;
use velo_core::radix_cache::TokenId;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        eprintln!("{}", USAGE);
        std::process::exit(1);
    });

    eprintln!("Loading model: {}", config.model.display());
    let t0 = Instant::now();
    let weights = load_gguf(&config.model).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    });

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
        eprintln!("No token ids provided; use --token-ids or --prompt");
        std::process::exit(1);
    }

    eprintln!(
        "Prompt: {} token(s): {:?}",
        prompt_tokens.len(),
        &prompt_tokens[..prompt_tokens.len().min(8)]
    );

    let t1 = Instant::now();
    let prompt_logits = model.forward_sequence(&prompt_tokens).unwrap_or_else(|e| {
        eprintln!("Forward pass failed: {e}");
        std::process::exit(1);
    });

    let ttft = t1.elapsed();

    let sampler = GreedySampler;
    let first_tok = sampler.sample(&TokenLogits::new(prompt_logits).unwrap());

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
        let logits = model.next_logits(&[last]).unwrap_or_else(|e| {
            eprintln!("Generation failed: {e}");
            std::process::exit(1);
        });
        let pred = sampler.sample(&logits);
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
}

// ── CLI parsing ───────────────────────────────────────────────────────────────

struct RunConfig {
    model: PathBuf,
    token_ids: Vec<TokenId>,
    max_tokens: usize,
}

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
}
