use std::time::Instant;

use velo_core::{
    EngineConfig, GreedyDraftModel, GreedyTargetModel, MemoryRuntimeConfig, MockBackend, Result,
    TokenId, VeloEngine,
};

fn main() -> Result<()> {
    let prompt = (0..128).collect::<Vec<_>>();
    let script = (0..2048).collect::<Vec<_>>();
    let mut engine = VeloEngine::new(EngineConfig {
        draft_window: 1,
        memory: MemoryRuntimeConfig::cpu(16, 16, 32, 1, 32),
        kv_type: velo_core::paged_attention::KvCacheType::Fp32,
    })
    .map_err(|error| velo_core::SpeculativeError::Model(error.to_string()))?;

    let cold = run_once(&mut engine, &script, &prompt, 128)?;
    let warm = run_once(&mut engine, &script, &prompt, 128)?;
    let rejection = run_with_rejection(&mut engine, &script, &prompt, 128)?;

    print_report("cold", &cold);
    print_report("warm", &warm);
    print_report("rejection", &rejection);

    Ok(())
}

#[derive(Debug)]
struct RunReport {
    generated_tokens: usize,
    elapsed_ms: f64,
    cache_hit_tokens: usize,
    cache_miss_tokens: usize,
    draft_calls: usize,
    target_calls: usize,
    accepted_tokens: usize,
    rejected_tokens: usize,
}

fn run_once(
    engine: &mut VeloEngine,
    script: &[TokenId],
    prompt: &[TokenId],
    max_new_tokens: usize,
) -> Result<RunReport> {
    let draft_backend = MockBackend::new(script.to_vec());
    let target_backend = MockBackend::new(script.to_vec());
    let mut draft = GreedyDraftModel::new(draft_backend);
    let mut target = GreedyTargetModel::new(target_backend);

    let started = Instant::now();
    let output = engine
        .generate(&mut draft, &mut target, prompt, max_new_tokens)
        .map_err(|error| velo_core::SpeculativeError::Model(error.to_string()))?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    Ok(RunReport {
        generated_tokens: output.tokens.len(),
        elapsed_ms,
        cache_hit_tokens: output.stats.cache_hit_tokens,
        cache_miss_tokens: output.stats.cache_miss_tokens,
        draft_calls: output.stats.speculative.draft_calls,
        target_calls: output.stats.speculative.target_calls,
        accepted_tokens: output.stats.speculative.accepted_tokens,
        rejected_tokens: output.stats.speculative.rejected_tokens,
    })
}

fn run_with_rejection(
    engine: &mut VeloEngine,
    script: &[TokenId],
    prompt: &[TokenId],
    max_new_tokens: usize,
) -> Result<RunReport> {
    let draft_backend = MockBackend::new(script.to_vec()).with_override(prompt.len() + 3, 99);
    let target_backend = MockBackend::new(script.to_vec());
    let mut draft = GreedyDraftModel::new(draft_backend);
    let mut target = GreedyTargetModel::new(target_backend);

    let started = Instant::now();
    let output = engine
        .generate(&mut draft, &mut target, prompt, max_new_tokens)
        .map_err(|error| velo_core::SpeculativeError::Model(error.to_string()))?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    Ok(RunReport {
        generated_tokens: output.tokens.len(),
        elapsed_ms,
        cache_hit_tokens: output.stats.cache_hit_tokens,
        cache_miss_tokens: output.stats.cache_miss_tokens,
        draft_calls: output.stats.speculative.draft_calls,
        target_calls: output.stats.speculative.target_calls,
        accepted_tokens: output.stats.speculative.accepted_tokens,
        rejected_tokens: output.stats.speculative.rejected_tokens,
    })
}

fn print_report(label: &str, report: &RunReport) {
    let tokens_per_second = report.generated_tokens as f64 / (report.elapsed_ms / 1000.0);

    println!("{label}:");
    println!("  generated_tokens: {}", report.generated_tokens);
    println!("  elapsed_ms: {:.3}", report.elapsed_ms);
    println!("  approx_tokens_per_second: {:.1}", tokens_per_second);
    println!("  cache_hit_tokens: {}", report.cache_hit_tokens);
    println!("  cache_miss_tokens: {}", report.cache_miss_tokens);
    println!("  draft_calls: {}", report.draft_calls);
    println!("  target_calls: {}", report.target_calls);
    println!("  accepted_tokens: {}", report.accepted_tokens);
    println!("  rejected_tokens: {}", report.rejected_tokens);
}
