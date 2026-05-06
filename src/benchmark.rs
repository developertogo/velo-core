use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::backend::{GreedyDraftModel, GreedyTargetModel};
use crate::engine::{EngineError, VeloEngine};
use crate::metal::Quantization;
use crate::mock_backend::MockBackend;
use crate::radix_cache::TokenId;
use crate::speculative::SpeculativeStats;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkMode {
    PromptProcessing,
    Generation,
    PromptPlusGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkFormat {
    Markdown,
    Csv,
    Json,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkConfig {
    pub mode: BenchmarkMode,
    pub prompt_len: usize,
    pub gen_len: usize,
    pub cached_depth: usize,
    pub repetitions: usize,
    pub warmups: usize,
    pub draft_window: usize,
    pub bytes_per_token: usize,
    pub page_tokens: usize,
    pub total_pages: usize,
    pub quantization: Quantization,
    pub model_name: String,
    pub backend_name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkSample {
    pub elapsed_ns: u128,
    pub ttft_ns: Option<u128>,
    pub tokens: usize,
    pub cache_hit_tokens: usize,
    pub cache_miss_tokens: usize,
    pub speculative: SpeculativeStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkRow {
    pub model_name: String,
    pub backend_name: String,
    pub test: String,
    pub mode: BenchmarkMode,
    pub prompt_len: usize,
    pub gen_len: usize,
    pub cached_depth: usize,
    pub repetitions: usize,
    pub warmups: usize,
    pub avg_ns: f64,
    pub stddev_ns: f64,
    pub avg_ts: f64,
    pub stddev_ts: f64,
    pub avg_ttft_ns: Option<f64>,
    pub avg_cache_hit_tokens: f64,
    pub avg_cache_miss_tokens: f64,
    pub avg_draft_calls: f64,
    pub avg_target_calls: f64,
    pub avg_accepted_tokens: f64,
    pub avg_rejected_tokens: f64,
    pub p50_ns: f64,
    pub p90_ns: f64,
    pub p99_ns: f64,
    pub baseline_avg_ts: Option<f64>,
    pub speedup_vs_baseline: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaBenchRow {
    pub model_filename: String,
    pub backend: String,
    pub test: String,
    pub n_prompt: usize,
    pub n_gen: usize,
    pub n_depth: usize,
    pub avg_ns: u128,
    pub stddev_ns: u128,
    pub avg_ts: f64,
    pub stddev_ts: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkReport {
    pub rows: Vec<BenchmarkRow>,
}

impl BenchmarkConfig {
    pub fn prompt_tokens(&self) -> Vec<TokenId> {
        (0..self.prompt_len as TokenId).collect()
    }

    pub fn script_tokens(&self) -> Vec<TokenId> {
        let len = self.prompt_len + self.gen_len + 16;
        (0..len as TokenId).collect()
    }
}

impl BenchmarkRow {
    pub fn tokens_processed(&self) -> usize {
        match self.mode {
            BenchmarkMode::PromptProcessing => self.prompt_len,
            BenchmarkMode::Generation => self.gen_len,
            BenchmarkMode::PromptPlusGeneration => self.prompt_len + self.gen_len,
        }
    }
}

pub fn run_benchmark(
    engine_config: &crate::engine::EngineConfig,
    config: &BenchmarkConfig,
) -> Result<BenchmarkReport, EngineError> {
    for _ in 0..config.warmups {
        let _ = run_single_case(engine_config, config)?;
    }

    let mut samples = Vec::with_capacity(config.repetitions);
    for _ in 0..config.repetitions {
        samples.push(run_single_case(engine_config, config)?);
    }

    Ok(BenchmarkReport {
        rows: vec![summarize(config, &samples)],
    })
}

pub fn run_single_case(
    engine_config: &crate::engine::EngineConfig,
    config: &BenchmarkConfig,
) -> Result<BenchmarkSample, EngineError> {
    let mut engine = VeloEngine::new(*engine_config)?;
    let prompt = config.prompt_tokens();
    let script = config.script_tokens();
    let prompt_slice = if config.cached_depth > 0 {
        &prompt[..config.cached_depth.min(prompt.len())]
    } else {
        &[]
    };

    if !prompt_slice.is_empty() {
        let _ = engine.prefill(prompt_slice)?;
    }

    let started = Instant::now();
    let (elapsed_ns, ttft_ns, output, prefill) = match config.mode {
        BenchmarkMode::PromptProcessing => {
            let prefill = engine.prefill(&prompt)?;
            (started.elapsed().as_nanos(), None, None, Some(prefill))
        }
        BenchmarkMode::Generation | BenchmarkMode::PromptPlusGeneration => {
            let mut draft = GreedyDraftModel::new(MockBackend::new(script.clone()));
            let mut target = GreedyTargetModel::new(MockBackend::new(script));
            let output = run_generation_with_ttft(
                &mut engine,
                &mut draft,
                &mut target,
                &prompt,
                config.gen_len,
            )?;
            (
                started.elapsed().as_nanos(),
                output.ttft_ns,
                Some(output.output),
                None,
            )
        }
    };

    let (tokens, cache_hit_tokens, cache_miss_tokens, speculative) = match config.mode {
        BenchmarkMode::PromptProcessing => {
            let prefill = prefill.expect("prefill output is always present");
            (
                config.prompt_len,
                prefill.stats.cache_hit_tokens,
                prefill.stats.cache_miss_tokens,
                prefill.stats.speculative,
            )
        }
        BenchmarkMode::Generation | BenchmarkMode::PromptPlusGeneration => {
            let output = output.expect("generation output is always present");
            let tokens = match config.mode {
                BenchmarkMode::Generation => config.gen_len,
                BenchmarkMode::PromptPlusGeneration => config.prompt_len + config.gen_len,
                BenchmarkMode::PromptProcessing => unreachable!(),
            };
            (
                tokens,
                output.stats.cache_hit_tokens,
                output.stats.cache_miss_tokens,
                output.stats.speculative,
            )
        }
    };

    Ok(BenchmarkSample {
        elapsed_ns,
        ttft_ns,
        tokens,
        cache_hit_tokens,
        cache_miss_tokens,
        speculative,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct GenerationTrace {
    ttft_ns: Option<u128>,
    output: crate::engine::EngineOutput,
}

fn run_generation_with_ttft<D, T>(
    engine: &mut VeloEngine,
    draft: &mut D,
    target: &mut T,
    prompt: &[TokenId],
    max_new_tokens: usize,
) -> Result<GenerationTrace, EngineError>
where
    D: crate::speculative::DraftModel,
    T: crate::speculative::TargetModel,
{
    let started = Instant::now();
    let cached_prefix = engine.prefill(prompt)?;
    draft.bind_prefix_cache(&cached_prefix.cached_prefix)?;
    target.bind_prefix_cache(&cached_prefix.cached_prefix)?;

    let mut session = engine.decoder().begin(prompt)?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut ttft_ns = None;

    while generated.len() < max_new_tokens {
        let remaining = max_new_tokens - generated.len();
        let drafted = session.draft(draft, target, remaining)?;

        if drafted.is_empty() && !session.has_pending_rejection() {
            break;
        }

        if ttft_ns.is_none() && !drafted.is_empty() {
            ttft_ns = Some(started.elapsed().as_nanos());
        }

        generated.extend_from_slice(&drafted);
        if !session.has_pending_rejection() {
            continue;
        }

        if let Some(token) = session.take_rejected_token() {
            if ttft_ns.is_none() {
                ttft_ns = Some(started.elapsed().as_nanos());
            }
            generated.push(token);
        }
    }

    let output = crate::engine::EngineOutput {
        tokens: generated,
        cached_prefix: cached_prefix.cached_prefix,
        cached_pages: cached_prefix.cached_pages,
        inserted_handle: None,
        inserted_pages: None,
        stats: crate::engine::EngineStats {
            cache_hit_tokens: cached_prefix.stats.cache_hit_tokens,
            cache_miss_tokens: cached_prefix.stats.cache_miss_tokens,
            speculative: *session.stats(),
        },
    };

    Ok(GenerationTrace { ttft_ns, output })
}

fn summarize(config: &BenchmarkConfig, samples: &[BenchmarkSample]) -> BenchmarkRow {
    let elapsed_ns = samples
        .iter()
        .map(|sample| sample.elapsed_ns as f64)
        .collect::<Vec<_>>();
    let tokens_per_second = samples
        .iter()
        .map(|sample| 1e9 * sample.tokens as f64 / sample.elapsed_ns as f64)
        .collect::<Vec<_>>();
    let ttft = samples
        .iter()
        .filter_map(|sample| sample.ttft_ns.map(|value| value as f64))
        .collect::<Vec<_>>();

    let mode_label = format_mode(
        config.mode,
        config.prompt_len,
        config.gen_len,
        config.cached_depth,
    );
    BenchmarkRow {
        model_name: config.model_name.clone(),
        backend_name: config.backend_name.clone(),
        test: mode_label,
        mode: config.mode,
        prompt_len: config.prompt_len,
        gen_len: config.gen_len,
        cached_depth: config.cached_depth,
        repetitions: config.repetitions,
        warmups: config.warmups,
        avg_ns: mean(&elapsed_ns),
        stddev_ns: stddev(&elapsed_ns),
        avg_ts: mean(&tokens_per_second),
        stddev_ts: stddev(&tokens_per_second),
        avg_ttft_ns: (!ttft.is_empty()).then(|| mean(&ttft)),
        avg_cache_hit_tokens: mean(
            &samples
                .iter()
                .map(|s| s.cache_hit_tokens as f64)
                .collect::<Vec<_>>(),
        ),
        avg_cache_miss_tokens: mean(
            &samples
                .iter()
                .map(|s| s.cache_miss_tokens as f64)
                .collect::<Vec<_>>(),
        ),
        avg_draft_calls: mean(
            &samples
                .iter()
                .map(|s| s.speculative.draft_calls as f64)
                .collect::<Vec<_>>(),
        ),
        avg_target_calls: mean(
            &samples
                .iter()
                .map(|s| s.speculative.target_calls as f64)
                .collect::<Vec<_>>(),
        ),
        avg_accepted_tokens: mean(
            &samples
                .iter()
                .map(|s| s.speculative.accepted_tokens as f64)
                .collect::<Vec<_>>(),
        ),
        avg_rejected_tokens: mean(
            &samples
                .iter()
                .map(|s| s.speculative.rejected_tokens as f64)
                .collect::<Vec<_>>(),
        ),
        p50_ns: percentile(&elapsed_ns, 50.0),
        p90_ns: percentile(&elapsed_ns, 90.0),
        p99_ns: percentile(&elapsed_ns, 99.0),
        baseline_avg_ts: None,
        speedup_vs_baseline: None,
    }
}

fn format_mode(
    mode: BenchmarkMode,
    prompt_len: usize,
    gen_len: usize,
    cached_depth: usize,
) -> String {
    let mut test = match mode {
        BenchmarkMode::PromptProcessing => format!("pp{prompt_len}"),
        BenchmarkMode::Generation => format!("tg{gen_len}"),
        BenchmarkMode::PromptPlusGeneration => format!("pp{prompt_len}+tg{gen_len}"),
    };

    if cached_depth > 0 {
        test.push_str(&format!(" @ d{cached_depth}"));
    }

    test
}

pub fn compare_with_llama_csv(
    rows: &mut [BenchmarkRow],
    llama_csv: &str,
) -> Result<Vec<LlamaBenchRow>, String> {
    let baseline_rows = parse_llama_csv(llama_csv)?;
    let baseline_by_test = baseline_rows
        .iter()
        .map(|row| (row.test.clone(), row))
        .collect::<BTreeMap<_, _>>();

    for row in rows {
        if let Some(baseline) = baseline_by_test.get(&row.test) {
            row.baseline_avg_ts = Some(baseline.avg_ts);
            row.speedup_vs_baseline = (baseline.avg_ts > 0.0).then(|| row.avg_ts / baseline.avg_ts);
        }
    }

    Ok(baseline_rows)
}

pub fn load_llama_csv(path: impl AsRef<Path>) -> Result<String, std::io::Error> {
    fs::read_to_string(path)
}

pub fn parse_llama_csv(input: &str) -> Result<Vec<LlamaBenchRow>, String> {
    let mut lines = input.lines().filter(|line| !line.trim().is_empty());
    let Some(header) = lines.next() else {
        return Ok(Vec::new());
    };

    let headers = parse_csv_line(header);
    let mut rows = Vec::new();

    for line in lines {
        let fields = parse_csv_line(line);
        let mut map = BTreeMap::new();
        for (header, value) in headers.iter().zip(fields.iter()) {
            map.insert(header.clone(), value.clone());
        }

        let model_filename = map.get("model_filename").cloned().unwrap_or_default();
        let backend = map.get("backends").cloned().unwrap_or_default();
        let n_prompt = parse_usize(map.get("n_prompt").map(String::as_str), "n_prompt")?;
        let n_gen = parse_usize(map.get("n_gen").map(String::as_str), "n_gen")?;
        let n_depth = parse_usize(map.get("n_depth").map(String::as_str), "n_depth")?;
        let avg_ns = parse_u128(map.get("avg_ns").map(String::as_str), "avg_ns")?;
        let stddev_ns = parse_u128(map.get("stddev_ns").map(String::as_str), "stddev_ns")?;
        let avg_ts = parse_f64(map.get("avg_ts").map(String::as_str), "avg_ts")?;
        let stddev_ts = parse_f64(map.get("stddev_ts").map(String::as_str), "stddev_ts")?;

        rows.push(LlamaBenchRow {
            model_filename,
            backend,
            test: format_llama_test(n_prompt, n_gen, n_depth),
            n_prompt,
            n_gen,
            n_depth,
            avg_ns,
            stddev_ns,
            avg_ts,
            stddev_ts,
        });
    }

    Ok(rows)
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes && chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = !in_quotes;
                }
            }
            ',' if !in_quotes => {
                fields.push(current);
                current = String::new();
            }
            _ => current.push(ch),
        }
    }

    fields.push(current);
    fields
}

fn parse_usize(value: Option<&str>, field: &str) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("missing {field}"))?
        .parse::<usize>()
        .map_err(|error| format!("failed to parse {field}: {error}"))
}

fn parse_u128(value: Option<&str>, field: &str) -> Result<u128, String> {
    value
        .ok_or_else(|| format!("missing {field}"))?
        .parse::<u128>()
        .map_err(|error| format!("failed to parse {field}: {error}"))
}

fn parse_f64(value: Option<&str>, field: &str) -> Result<f64, String> {
    value
        .ok_or_else(|| format!("missing {field}"))?
        .parse::<f64>()
        .map_err(|error| format!("failed to parse {field}: {error}"))
}

fn format_llama_test(n_prompt: usize, n_gen: usize, n_depth: usize) -> String {
    let mut test = if n_prompt > 0 && n_gen == 0 {
        format!("pp{n_prompt}")
    } else if n_gen > 0 && n_prompt == 0 {
        format!("tg{n_gen}")
    } else {
        format!("pp{n_prompt}+tg{n_gen}")
    };

    if n_depth > 0 {
        test.push_str(&format!(" @ d{n_depth}"));
    }

    test
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
}

impl std::fmt::Display for BenchmarkMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PromptProcessing => formatter.write_str("prompt-processing"),
            Self::Generation => formatter.write_str("generation"),
            Self::PromptPlusGeneration => formatter.write_str("prompt-plus-generation"),
        }
    }
}

impl std::fmt::Display for BenchmarkFormat {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Markdown => formatter.write_str("markdown"),
            Self::Csv => formatter.write_str("csv"),
            Self::Json => formatter.write_str("json"),
        }
    }
}

impl BenchmarkReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "| test | model | backend | t/s | ttft ns | p50 ns | p90 ns | cache hit | speedup |\n",
        );
        out.push_str("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for row in &self.rows {
            let ttft = row
                .avg_ttft_ns
                .map(|value| format!("{value:.0}"))
                .unwrap_or_else(|| "-".to_string());
            let speedup = row
                .speedup_vs_baseline
                .map(|value| format!("{value:.2}x"))
                .unwrap_or_else(|| "-".to_string());
            out.push_str(&format!(
                "| {} | {} | {} | {:.2} | {} | {:.0} | {:.0} | {:.2} | {} |\n",
                row.test,
                row.model_name,
                row.backend_name,
                row.avg_ts,
                ttft,
                row.p50_ns,
                row.p90_ns,
                row.avg_cache_hit_tokens,
                speedup
            ));
        }
        out
    }

    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        out.push_str("model_name,backend_name,test,mode,prompt_len,gen_len,cached_depth,repetitions,warmups,avg_ns,stddev_ns,p50_ns,p90_ns,p99_ns,avg_ts,stddev_ts,avg_ttft_ns,avg_cache_hit_tokens,avg_cache_miss_tokens,avg_draft_calls,avg_target_calls,avg_accepted_tokens,avg_rejected_tokens,baseline_avg_ts,speedup_vs_baseline\n");
        for row in &self.rows {
            let ttft = row
                .avg_ttft_ns
                .map_or(String::new(), |value| value.to_string());
            let baseline = row
                .baseline_avg_ts
                .map_or(String::new(), |value| value.to_string());
            let speedup = row
                .speedup_vs_baseline
                .map_or(String::new(), |value| value.to_string());
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{:.0},{:.6},{:.0},{:.0},{:.0},{:.6},{:.6},{},{},{},{},{},{},{},{},{}\n",
                row.model_name,
                row.backend_name,
                row.test,
                row.mode,
                row.prompt_len,
                row.gen_len,
                row.cached_depth,
                row.repetitions,
                row.warmups,
                row.avg_ns,
                row.stddev_ns,
                row.p50_ns,
                row.p90_ns,
                row.p99_ns,
                row.avg_ts,
                row.stddev_ts,
                ttft,
                row.avg_cache_hit_tokens,
                row.avg_cache_miss_tokens,
                row.avg_draft_calls,
                row.avg_target_calls,
                row.avg_accepted_tokens,
                row.avg_rejected_tokens,
                baseline,
                speedup
            ));
        }
        out
    }

    pub fn to_json(&self) -> String {
        let mut out = String::from("[");
        for (index, row) in self.rows.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"model_name\":\"{}\",\"backend_name\":\"{}\",\"test\":\"{}\",\"mode\":\"{}\",\"prompt_len\":{},\"gen_len\":{},\"cached_depth\":{},\"repetitions\":{},\"warmups\":{},\"avg_ns\":{:.0},\"stddev_ns\":{:.6},\"p50_ns\":{:.0},\"p90_ns\":{:.0},\"p99_ns\":{:.0},\"avg_ts\":{:.6},\"stddev_ts\":{:.6},\"avg_ttft_ns\":{},\"avg_cache_hit_tokens\":{:.6},\"avg_cache_miss_tokens\":{:.6},\"avg_draft_calls\":{:.6},\"avg_target_calls\":{:.6},\"avg_accepted_tokens\":{:.6},\"avg_rejected_tokens\":{:.6},\"baseline_avg_ts\":{},\"speedup_vs_baseline\":{}}}",
                row.model_name,
                row.backend_name,
                row.test,
                row.mode,
                row.prompt_len,
                row.gen_len,
                row.cached_depth,
                row.repetitions,
                row.warmups,
                row.avg_ns,
                row.stddev_ns,
                row.p50_ns,
                row.p90_ns,
                row.p99_ns,
                row.avg_ts,
                row.stddev_ts,
                row.avg_ttft_ns.map_or_else(|| "null".to_string(), |value| value.to_string()),
                row.avg_cache_hit_tokens,
                row.avg_cache_miss_tokens,
                row.avg_draft_calls,
                row.avg_target_calls,
                row.avg_accepted_tokens,
                row.avg_rejected_tokens,
                row.baseline_avg_ts.map_or_else(|| "null".to_string(), |value| value.to_string()),
                row.speedup_vs_baseline.map_or_else(|| "null".to_string(), |value| value.to_string()),
            ));
        }
        out.push(']');
        out
    }
}

impl std::fmt::Display for LlamaBenchRow {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} | {} | {} | {:.2} t/s",
            self.test, self.model_filename, self.backend, self.avg_ts
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::MemoryRuntimeConfig;

    #[test]
    fn parses_llama_csv_rows() {
        let csv = "model_filename,backends,n_prompt,n_gen,n_depth,avg_ns,stddev_ns,avg_ts,stddev_ts\n\
                   llama-3-8b,metal,512,128,0,1000000,500,128.0,0.5";
        let rows = parse_llama_csv(csv).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model_filename, "llama-3-8b");
        assert_eq!(rows[0].avg_ts, 128.0);
    }

    #[test]
    fn benchmark_report_markdown_not_empty() {
        let row = BenchmarkRow {
            model_name: "test".to_string(),
            backend_name: "test".to_string(),
            test: "test".to_string(),
            mode: BenchmarkMode::Generation,
            prompt_len: 1,
            gen_len: 1,
            cached_depth: 0,
            repetitions: 1,
            warmups: 0,
            avg_ns: 1000.0,
            stddev_ns: 0.0,
            avg_ts: 1000.0,
            stddev_ts: 0.0,
            avg_ttft_ns: None,
            avg_cache_hit_tokens: 0.0,
            avg_cache_miss_tokens: 1.0,
            avg_draft_calls: 1.0,
            avg_target_calls: 1.0,
            avg_accepted_tokens: 1.0,
            avg_rejected_tokens: 0.0,
            p50_ns: 1000.0,
            p90_ns: 1000.0,
            p99_ns: 1000.0,
            baseline_avg_ts: None,
            speedup_vs_baseline: None,
        };
        let report = BenchmarkReport { rows: vec![row] };
        let md = report.to_markdown();
        assert!(md.contains("test"));
        assert!(md.contains("|"));
    }

    #[test]
    fn benchmark_report_json_parsable() {
        let row = BenchmarkRow {
            model_name: "test".to_string(),
            backend_name: "test".to_string(),
            test: "test".to_string(),
            mode: BenchmarkMode::Generation,
            prompt_len: 1,
            gen_len: 1,
            cached_depth: 0,
            repetitions: 1,
            warmups: 0,
            avg_ns: 1000.0,
            stddev_ns: 0.0,
            avg_ts: 1000.0,
            stddev_ts: 0.0,
            avg_ttft_ns: None,
            avg_cache_hit_tokens: 0.0,
            avg_cache_miss_tokens: 1.0,
            avg_draft_calls: 1.0,
            avg_target_calls: 1.0,
            avg_accepted_tokens: 1.0,
            avg_rejected_tokens: 0.0,
            p50_ns: 1000.0,
            p90_ns: 1000.0,
            p99_ns: 1000.0,
            baseline_avg_ts: None,
            speedup_vs_baseline: None,
        };
        let report = BenchmarkReport { rows: vec![row] };
        let json = report.to_json();
        assert!(json.contains("\"test\""));
    }

    #[test]
    fn formats_test_name_like_llama_bench() {
        assert_eq!(
            format_mode(BenchmarkMode::PromptProcessing, 512, 0, 0),
            "pp512"
        );
        assert_eq!(format_mode(BenchmarkMode::Generation, 0, 128, 0), "tg128");
        assert_eq!(
            format_mode(BenchmarkMode::PromptPlusGeneration, 512, 128, 64),
            "pp512+tg128 @ d64"
        );
    }

    #[test]
    fn calculates_percentiles() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 50.0), 3.0);
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 100.0), 5.0);
    }

    #[test]
    fn produces_comparison_with_baseline() {
        let mut rows = vec![BenchmarkRow {
            model_name: "velo".into(),
            backend_name: "mock".into(),
            test: "pp512".into(),
            mode: BenchmarkMode::PromptProcessing,
            prompt_len: 512,
            gen_len: 0,
            cached_depth: 0,
            repetitions: 5,
            warmups: 1,
            avg_ns: 10.0,
            stddev_ns: 0.0,
            avg_ts: 100.0,
            stddev_ts: 0.0,
            avg_ttft_ns: None,
            avg_cache_hit_tokens: 0.0,
            avg_cache_miss_tokens: 512.0,
            avg_draft_calls: 0.0,
            avg_target_calls: 0.0,
            avg_accepted_tokens: 0.0,
            avg_rejected_tokens: 0.0,
            p50_ns: 1000.0,
            p90_ns: 1000.0,
            p99_ns: 1000.0,
            baseline_avg_ts: None,
            speedup_vs_baseline: None,
        }];

        let csv = "build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,split_mode,main_gpu,no_kv_offload,flash_attn,tensor_split,use_mmap,embeddings,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts\n\
                   \"a\",\"1\",\"cpu\",\"gpu\",\"CUDA\",\"model.gguf\",\"model\",\"1\",\"1\",\"1\",\"1\",\"1\",\"0x0\",\"0\",\"50\",\"f16\",\"f16\",\"0\",\"layer\",\"0\",\"0\",\"0\",\"0.00\",\"1\",\"0\",\"512\",\"0\",\"0\",\"2025-04-24T11:57:09Z\",\"70285660\",\"982040\",\"50.0\",\"1.0\"";
        compare_with_llama_csv(&mut rows, csv).unwrap();

        assert_eq!(rows[0].baseline_avg_ts, Some(50.0));
        assert_eq!(rows[0].speedup_vs_baseline, Some(2.0));
    }

    #[test]
    fn test_benchmark_sample_prompt() {
        let engine_config = crate::engine::EngineConfig {
            draft_window: 1,
            memory: MemoryRuntimeConfig::cpu(16, 16, 32, 1, 32),
            kv_type: crate::paged_attention::KvCacheType::Fp32,
        };

        let config = BenchmarkConfig {
            mode: BenchmarkMode::PromptProcessing,
            prompt_len: 8,
            gen_len: 0,
            cached_depth: 0,
            repetitions: 1,
            warmups: 0,
            draft_window: 1,
            bytes_per_token: 0,
            page_tokens: 16,
            total_pages: 32,
            quantization: crate::metal::Quantization::F32,
            model_name: "test".into(),
            backend_name: "cpu".into(),
        };

        let sample = run_single_case(&engine_config, &config).unwrap();
        assert_eq!(sample.tokens, 8);
        assert!(sample.elapsed_ns > 0);
    }

    #[test]
    fn benchmark_row_tokens_processed() {
        let mut row = BenchmarkRow {
            model_name: "".into(),
            backend_name: "".into(),
            test: "".into(),
            mode: BenchmarkMode::PromptProcessing,
            prompt_len: 10,
            gen_len: 20,
            cached_depth: 0,
            repetitions: 1,
            warmups: 0,
            avg_ns: 0.0,
            stddev_ns: 0.0,
            avg_ts: 0.0,
            stddev_ts: 0.0,
            avg_ttft_ns: None,
            avg_cache_hit_tokens: 0.0,
            avg_cache_miss_tokens: 0.0,
            avg_draft_calls: 0.0,
            avg_target_calls: 0.0,
            avg_accepted_tokens: 0.0,
            avg_rejected_tokens: 0.0,
            p50_ns: 1000.0,
            p90_ns: 1000.0,
            p99_ns: 1000.0,
            baseline_avg_ts: None,
            speedup_vs_baseline: None,
        };
        assert_eq!(row.tokens_processed(), 10);
        row.mode = BenchmarkMode::Generation;
        assert_eq!(row.tokens_processed(), 20);
        row.mode = BenchmarkMode::PromptPlusGeneration;
        assert_eq!(row.tokens_processed(), 30);
    }

    #[test]
    fn benchmark_report_csv_not_empty() {
        let row = BenchmarkRow {
            model_name: "m".into(),
            backend_name: "b".into(),
            test: "t".into(),
            mode: BenchmarkMode::Generation,
            prompt_len: 1,
            gen_len: 1,
            cached_depth: 0,
            repetitions: 1,
            warmups: 0,
            avg_ns: 1000.0,
            stddev_ns: 0.0,
            avg_ts: 1000.0,
            stddev_ts: 0.0,
            avg_ttft_ns: Some(500.0),
            avg_cache_hit_tokens: 0.0,
            avg_cache_miss_tokens: 1.0,
            avg_draft_calls: 1.0,
            avg_target_calls: 1.0,
            avg_accepted_tokens: 1.0,
            avg_rejected_tokens: 0.0,
            p50_ns: 1000.0,
            p90_ns: 1000.0,
            p99_ns: 1000.0,
            baseline_avg_ts: Some(800.0),
            speedup_vs_baseline: Some(1.25),
        };
        let report = BenchmarkReport { rows: vec![row] };
        let csv = report.to_csv();
        assert!(csv.contains("m,b,t"));
        assert!(csv.contains("1.25"));
    }

    #[test]
    fn mean_stddev_edge_cases() {
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(stddev(&[1.0]), 0.0);
        assert_eq!(mean(&[1.0, 2.0]), 1.5);
    }

    #[test]
    fn parse_llama_csv_edge_cases() {
        assert!(parse_llama_csv("").unwrap().is_empty());
        assert!(parse_llama_csv("h1,h2\nv1,v2").is_err()); // Missing expected headers
    }

    #[test]
    fn benchmark_enum_displays() {
        assert_eq!(
            format!("{}", BenchmarkMode::PromptProcessing),
            "prompt-processing"
        );
        assert_eq!(format!("{}", BenchmarkFormat::Json), "json");
    }

    #[test]
    fn llama_bench_row_display() {
        let row = LlamaBenchRow {
            model_filename: "f".into(),
            backend: "b".into(),
            test: "t".into(),
            n_prompt: 0,
            n_gen: 0,
            n_depth: 0,
            avg_ns: 0,
            stddev_ns: 0,
            avg_ts: 10.0,
            stddev_ts: 0.0,
        };
        assert!(format!("{}", row).contains("10.00 t/s"));
    }

    #[test]
    fn run_benchmark_warmup_and_reps() {
        let engine_config = crate::engine::EngineConfig {
            draft_window: 1,
            memory: MemoryRuntimeConfig::cpu(16, 16, 32, 1, 32),
            kv_type: crate::paged_attention::KvCacheType::Fp32,
        };
        let config = BenchmarkConfig {
            mode: BenchmarkMode::PromptProcessing,
            prompt_len: 4,
            gen_len: 0,
            cached_depth: 0,
            repetitions: 2,
            warmups: 1,
            draft_window: 1,
            bytes_per_token: 0,
            page_tokens: 16,
            total_pages: 32,
            quantization: crate::metal::Quantization::F32,
            model_name: "test".into(),
            backend_name: "cpu".into(),
        };
        let report = run_benchmark(&engine_config, &config).unwrap();
        assert_eq!(report.rows.len(), 1);
        assert_eq!(report.rows[0].repetitions, 2);
    }

    #[test]
    fn test_benchmark_sample_generation() {
        let engine_config = crate::engine::EngineConfig {
            draft_window: 1,
            memory: MemoryRuntimeConfig::cpu(16, 16, 32, 1, 32),
            kv_type: crate::paged_attention::KvCacheType::Fp32,
        };

        let config = BenchmarkConfig {
            mode: BenchmarkMode::Generation,
            prompt_len: 4,
            gen_len: 4,
            cached_depth: 0,
            repetitions: 1,
            warmups: 0,
            draft_window: 1,
            bytes_per_token: 0,
            page_tokens: 16,
            total_pages: 32,
            quantization: crate::metal::Quantization::F32,
            model_name: "test".into(),
            backend_name: "cpu".into(),
        };

        let sample = run_single_case(&engine_config, &config).unwrap();
        assert_eq!(sample.tokens, 4);
        assert!(sample.elapsed_ns > 0);
    }

    #[test]
    fn test_benchmark_sample_prompt_plus_generation() {
        let engine_config = crate::engine::EngineConfig {
            draft_window: 1,
            memory: MemoryRuntimeConfig::cpu(16, 16, 32, 1, 32),
            kv_type: crate::paged_attention::KvCacheType::Fp32,
        };

        let config = BenchmarkConfig {
            mode: BenchmarkMode::PromptPlusGeneration,
            prompt_len: 4,
            gen_len: 4,
            cached_depth: 2,
            repetitions: 1,
            warmups: 0,
            draft_window: 1,
            bytes_per_token: 0,
            page_tokens: 16,
            total_pages: 32,
            quantization: crate::metal::Quantization::F32,
            model_name: "test".into(),
            backend_name: "cpu".into(),
        };

        let sample = run_single_case(&engine_config, &config).unwrap();
        assert_eq!(sample.tokens, 8);
        assert!(sample.elapsed_ns > 0);
    }

    #[test]
    fn test_benchmark_helpers() {
        assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(mean(&[]), 0.0);
        assert!(stddev(&[1.0, 2.0, 3.0]) > 0.0);
        assert_eq!(stddev(&[1.0]), 0.0);

        assert_eq!(format_llama_test(512, 0, 0), "pp512");
        assert_eq!(format_llama_test(0, 128, 0), "tg128");
        assert_eq!(format_llama_test(512, 128, 0), "pp512+tg128");
        assert_eq!(format_llama_test(512, 0, 256), "pp512 @ d256");

        assert_eq!(
            format!("{}", BenchmarkMode::PromptProcessing),
            "prompt-processing"
        );
        assert_eq!(format!("{}", BenchmarkFormat::Json), "json");
    }

    #[test]
    fn test_parse_csv_line() {
        assert_eq!(parse_csv_line("a,b,c"), vec!["a", "b", "c"]);
        assert_eq!(parse_csv_line("\"a,b\",c"), vec!["a,b", "c"]);
        assert_eq!(parse_csv_line("\"a\"\"b\",c"), vec!["a\"b", "c"]);
    }

    #[test]
    fn test_benchmark_report_csv() {
        let row = BenchmarkRow {
            model_name: "m".into(),
            backend_name: "b".into(),
            test: "t".into(),
            mode: BenchmarkMode::PromptProcessing,
            prompt_len: 1,
            gen_len: 0,
            cached_depth: 0,
            repetitions: 1,
            warmups: 0,
            avg_ns: 100.0,
            stddev_ns: 1.0,
            avg_ts: 10.0,
            stddev_ts: 0.1,
            avg_ttft_ns: Some(50.0),
            avg_cache_hit_tokens: 0.0,
            avg_cache_miss_tokens: 1.0,
            avg_draft_calls: 0.0,
            avg_target_calls: 0.0,
            avg_accepted_tokens: 0.0,
            avg_rejected_tokens: 0.0,
            p50_ns: 100.0,
            p90_ns: 100.0,
            p99_ns: 100.0,
            baseline_avg_ts: Some(5.0),
            speedup_vs_baseline: Some(2.0),
        };
        let report = BenchmarkReport { rows: vec![row] };
        let csv = report.to_csv();
        assert!(csv.contains("m,b,t,prompt-processing"));
        assert!(csv.contains("50,0,1,0,0,0,0,5,2"));
    }
}
