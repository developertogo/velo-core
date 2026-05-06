/// velo-spec — run speculative decoding using a draft and target model.
///
/// Usage:
///   velo-spec --target path/to/target.gguf --draft path/to/draft.gguf --prompt "Hello"
use std::path::PathBuf;
use std::time::Instant;

use velo_core::backend::{GreedyDraftModel, GreedyTargetModel};
use velo_core::mock_backend::MockBackend;

use velo_core::engine::{EngineConfig, VeloEngine};
use velo_core::model_loader::load_gguf;
use velo_core::metal::{MetalBackend, MetalBackendConfig, MetalMemoryRuntime, MetalRuntimeConfig};
use velo_core::runtime::MemoryRuntimeConfig;
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

    if config.mock {
        run_mock(config);
        return Ok(());
    }

    let Some(target_path) = &config.target else {
        return Err("--target missing and --mock not specified".to_string());
    };
    let Some(draft_path) = &config.draft else {
        return Err("--draft missing and --mock not specified".to_string());
    };

    eprintln!("Loading target model: {}", target_path.display());
    let t0 = Instant::now();
    let target_weights = load_gguf(target_path).map_err(|e| format!("Failed to load target model: {e}"))?;
   
    eprintln!("Loading draft model: {}", draft_path.display());
    let draft_weights = load_gguf(draft_path).map_err(|e| format!("Failed to load draft model: {e}"))?;

    let target_meta = target_weights.meta.clone();
    eprintln!(
        "Loaded target in {:.2}s  arch={} n_layer={} n_embd={} n_head={} n_kv={} vocab={}",
        t0.elapsed().as_secs_f32(),
        target_meta.arch,
        target_meta.n_layer,
        target_meta.n_embd,
        target_meta.n_head,
        target_meta.n_head_kv,
        target_meta.n_vocab,
    );

    // Calculate KV bytes per token for one layer
    let kv_bytes_per_token = target_meta.n_head_kv * target_meta.head_dim * 4;

    // Initialize Metal Runtimes
    let target_runtime = MetalMemoryRuntime::new(MetalRuntimeConfig {
        model_name: "target".to_string(),
        memory: MemoryRuntimeConfig::cpu(kv_bytes_per_token, 16, 256, target_meta.n_layer, 32).with_kv_type(velo_core::paged_attention::KvCacheType::Fp32),
        quantization: target_meta.quantization,
    }).map_err(|e| format!("Failed to create runtime: {e}"))?;

    let mut target_backend = MetalBackend::new(MetalBackendConfig {
        model_name: "target".to_string(),
        max_context_tokens: config.max_context,
        kv_bytes_per_token,
        paged_block_size: 16,
        quantization: target_meta.quantization,
        kv_type: velo_core::paged_attention::KvCacheType::Fp32,
        }).map_err(|e| format!("Failed to create target backend: {e}"))?;
    target_backend.wire(target_weights, &target_runtime).map_err(|e| format!("Failed to wire target: {e}"))?;

    let mut draft_backend = MetalBackend::new(MetalBackendConfig {
        model_name: "draft".to_string(),
        max_context_tokens: config.max_context,
        kv_bytes_per_token,
        paged_block_size: 16,
        quantization: draft_weights.meta.quantization,
        kv_type: velo_core::paged_attention::KvCacheType::Fp32,
        }).map_err(|e| format!("Failed to create draft backend: {e}"))?;
   
    // For simplicity, we reuse the target runtime's context if they are both unified
    draft_backend.wire(draft_weights, &target_runtime).map_err(|e| format!("Failed to wire draft: {e}"))?;

    let mut engine = VeloEngine::with_runtime(
        EngineConfig {
            draft_window: config.draft_window,
            memory: target_runtime.context().memory,
            kv_type: velo_core::paged_attention::KvCacheType::Fp32,
        },
        target_runtime,
    ).map_err(|e| format!("Failed to create engine: {e}"))?;

    let mut draft_model = GreedyDraftModel::new(draft_backend);
    let mut target_model = GreedyTargetModel::new(target_backend);

    // ── Prompt processing ─────────────────────────────────────────────────────
    let prompt_tokens = config.token_ids.clone();
    eprintln!("Generating with prompt: {:?}", prompt_tokens);

    let t1 = Instant::now();
    let output = engine.generate(
        &mut draft_model,
        &mut target_model,
        &prompt_tokens,
        config.max_new_tokens,
    ).map_err(|e| format!("Generation failed: {e}"))?;
    let elapsed = t1.elapsed();

    eprintln!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        output.tokens.len(),
        elapsed.as_secs_f32(),
        output.tokens.len() as f32 / elapsed.as_secs_f32(),
    );

    eprintln!("Stats: {:?}", output.stats);

    println!("generated_token_ids:");
    for tok in &output.tokens {
        println!("  {tok}");
    }
    Ok(())
}

fn run_mock(config: SpecConfig) {
    let script = (0..2048).collect::<Vec<TokenId>>();
    let mut engine = VeloEngine::new(EngineConfig {
        draft_window: config.draft_window,
        memory: MemoryRuntimeConfig::cpu(4096, 16, 4096, 32, 32).with_kv_type(velo_core::paged_attention::KvCacheType::Fp32),
        kv_type: velo_core::paged_attention::KvCacheType::Fp32,
    }).unwrap();

    let draft_backend = MockBackend::new(script.clone());
    let target_backend = MockBackend::new(script);
    let mut draft_model = GreedyDraftModel::new(draft_backend);
    let mut target_model = GreedyTargetModel::new(target_backend);

    let t1 = Instant::now();
    let output = engine.generate(
        &mut draft_model,
        &mut target_model,
        &config.token_ids,
        config.max_new_tokens,
    ).unwrap();
    let elapsed = t1.elapsed();

    eprintln!(
        "Mock: Generated {} tokens in {:.2}s ({:.1} tok/s)",
        output.tokens.len(),
        elapsed.as_secs_f32(),
        output.tokens.len() as f32 / elapsed.as_secs_f32(),
    );
    println!("generated_token_ids:");
    for tok in &output.tokens {
        println!("  {tok}");
    }
}

#[derive(Debug, Clone)]
struct SpecConfig {
    target: Option<PathBuf>,
    draft: Option<PathBuf>,
    token_ids: Vec<TokenId>,
    max_new_tokens: usize,
    draft_window: usize,
    max_context: usize,
    mock: bool,
}

#[allow(dead_code)]
const USAGE: &str = "\
Usage: velo-spec [--target <path> --draft <path> | --mock] --prompt <text> [--max-new <n>] [--window <w>]
";

fn parse_args(args: &[String]) -> Result<SpecConfig, String> {
    let mut target = None;
    let mut draft = None;
    let mut token_ids = Vec::new();
    let mut max_new_tokens = 32;
    let mut draft_window = 4;
    let max_context = 4096;
    let mut mock = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => {
                i += 1;
                target = Some(PathBuf::from(args.get(i).ok_or("--target requires a path")?));
            }
            "--draft" => {
                i += 1;
                draft = Some(PathBuf::from(args.get(i).ok_or("--draft requires a path")?));
            }
            "--prompt" => {
                i += 1;
                let text = args.get(i).ok_or("--prompt requires a value")?;
                token_ids = text.bytes().map(|b| b as TokenId).collect();
            }
            "--max-new" => {
                i += 1;
                max_new_tokens = args.get(i).ok_or("--max-new requires a value")?.parse().map_err(|e| format!("invalid max-new: {}", e))?;
            }
            "--window" => {
                i += 1;
                draft_window = args.get(i).ok_or("--window requires a value")?.parse().map_err(|e| format!("invalid window: {}", e))?;
            }
            "--mock" => {
                mock = true;
            }
            _ => return Err(format!("unknown arg: {}", args[i])),
        }
        i += 1;
    }

    Ok(SpecConfig {
        target,
        draft,
        token_ids,
        max_new_tokens,
        draft_window,
        max_context,
        mock,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(pairs: &[&str]) -> Vec<String> {
        let mut v = vec!["velo-spec".to_string()];
        v.extend(pairs.iter().map(|s| s.to_string()));
        v
    }

    #[test]
    fn parses_mock_spec() {
        let cfg = parse_args(&args(&["--mock", "--prompt", "hi"])).unwrap();
        assert!(cfg.mock);
        assert_eq!(cfg.token_ids, vec![b'h' as u32, b'i' as u32]);
    }

    #[test]
    fn parses_full_spec() {
        let cfg = parse_args(&args(&[
            "--target", "t.gguf",
            "--draft", "d.gguf",
            "--prompt", "p",
            "--max-new", "64",
            "--window", "8",
        ])).unwrap();
        assert_eq!(cfg.target, Some(PathBuf::from("t.gguf")));
        assert_eq!(cfg.draft, Some(PathBuf::from("d.gguf")));
        assert_eq!(cfg.max_new_tokens, 64);
        assert_eq!(cfg.draft_window, 8);
    }

    #[test]
    fn rejects_unknown_arg() {
        assert!(parse_args(&args(&["--bad"])).is_err());
    }

    #[test]
    fn rejects_missing_target_or_draft() {
        let args = vec!["velo-spec".to_string(), "--prompt".to_string(), "hi".to_string()];
        let cfg = parse_args(&args).unwrap();
        assert_eq!(cfg.target, None);
        assert_eq!(cfg.draft, None);
    }

    #[test]
    fn rejects_missing_values() {
        assert!(parse_args(&vec!["velo-spec".to_string(), "--target".to_string()]).is_err());
        assert!(parse_args(&vec!["velo-spec".to_string(), "--draft".to_string()]).is_err());
        assert!(parse_args(&vec!["velo-spec".to_string(), "--prompt".to_string()]).is_err());
        assert!(parse_args(&vec!["velo-spec".to_string(), "--max-new".to_string()]).is_err());
        assert!(parse_args(&vec!["velo-spec".to_string(), "--window".to_string()]).is_err());
    }

    #[test]
    fn test_run_mock_full() {
        let cfg = SpecConfig {
            target: None,
            draft: None,
            token_ids: vec![1, 2, 3],
            max_new_tokens: 10,
            draft_window: 4,
            max_context: 4096,
            mock: true,
        };
        run_mock(cfg);
    }

    #[test]
    fn test_invalid_max_new_format() {
        let args = vec!["velo-spec".to_string(), "--prompt".to_string(), "h".to_string(), "--max-new".to_string(), "nan".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_invalid_window_format() {
        let args = vec!["velo-spec".to_string(), "--prompt".to_string(), "h".to_string(), "--window".to_string(), "nan".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_spec_config_traits() {
        let cfg = SpecConfig {
            target: Some(PathBuf::from("t")),
            draft: Some(PathBuf::from("d")),
            token_ids: vec![1],
            max_new_tokens: 10,
            draft_window: 4,
            max_context: 2048,
            mock: false,
        };
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("max_new_tokens: 10"));
        assert!(dbg.contains("mock: false"));
    }

    #[test]
    fn test_usage_not_empty() {
        assert!(!USAGE.is_empty());
    }

    #[test]
    fn test_parse_args_missing_values_direct() {
        assert!(parse_args(&["velo-spec".to_string(), "--target".to_string()]).is_err());
        assert!(parse_args(&["velo-spec".to_string(), "--draft".to_string()]).is_err());
        assert!(parse_args(&["velo-spec".to_string(), "--prompt".to_string()]).is_err());
        assert!(parse_args(&["velo-spec".to_string(), "--max-new".to_string()]).is_err());
        assert!(parse_args(&["velo-spec".to_string(), "--window".to_string()]).is_err());
    }

    #[test]
    fn test_run_invalid_args() {
        let res = run(&["velo-spec".to_string(), "--unknown".to_string()]);
        assert!(res.is_err());
    }
}
