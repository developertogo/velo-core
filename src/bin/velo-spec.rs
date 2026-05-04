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
    let config = parse_args(&args).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        eprintln!("{}", USAGE);
        std::process::exit(1);
    });

    if config.mock {
        run_mock(config);
        return;
    }

    let Some(target_path) = &config.target else {
        eprintln!("Error: --target missing and --mock not specified");
        std::process::exit(1);
    };
    let Some(draft_path) = &config.draft else {
        eprintln!("Error: --draft missing and --mock not specified");
        std::process::exit(1);
    };

    eprintln!("Loading target model: {}", target_path.display());
    let t0 = Instant::now();
    let target_weights = load_gguf(target_path).unwrap_or_else(|e| {
        eprintln!("Failed to load target model: {e}");
        std::process::exit(1);
    });
    
    eprintln!("Loading draft model: {}", draft_path.display());
    let draft_weights = load_gguf(draft_path).unwrap_or_else(|e| {
        eprintln!("Failed to load draft model: {e}");
        std::process::exit(1);
    });

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
        memory: MemoryRuntimeConfig::cpu(kv_bytes_per_token, 16, 256, target_meta.n_layer, 32),
        quantization: target_meta.quantization,
    }).unwrap();

    let mut target_backend = MetalBackend::new(MetalBackendConfig {
        model_name: "target".to_string(),
        max_context_tokens: config.max_context,
        kv_bytes_per_token,
        paged_block_size: 16,
        quantization: target_meta.quantization,
    }).unwrap();
    target_backend.wire(target_weights, &target_runtime).unwrap();

    let mut draft_backend = MetalBackend::new(MetalBackendConfig {
        model_name: "draft".to_string(),
        max_context_tokens: config.max_context,
        kv_bytes_per_token,
        paged_block_size: 16,
        quantization: draft_weights.meta.quantization,
    }).unwrap();
    
    // For simplicity, we reuse the target runtime's context if they are both unified
    draft_backend.wire(draft_weights, &target_runtime).unwrap();

    let mut engine = VeloEngine::with_runtime(
        EngineConfig {
            draft_window: config.draft_window,
            memory: target_runtime.context().memory,
        },
        target_runtime,
    ).unwrap();

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
    ).unwrap();
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
}

fn run_mock(config: SpecConfig) {
    let script = (0..2048).collect::<Vec<TokenId>>();
    let mut engine = VeloEngine::new(EngineConfig {
        draft_window: config.draft_window,
        memory: MemoryRuntimeConfig::cpu(4096, 16, 4096, 32, 32),
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

struct SpecConfig {
    target: Option<PathBuf>,
    draft: Option<PathBuf>,
    token_ids: Vec<TokenId>,
    max_new_tokens: usize,
    draft_window: usize,
    max_context: usize,
    mock: bool,
}

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
                max_new_tokens = args.get(i).ok_or("--max-new requires a value")?.parse().unwrap();
            }
            "--window" => {
                i += 1;
                draft_window = args.get(i).ok_or("--window requires a value")?.parse().unwrap();
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
