use std::env;
use std::error::Error;
use std::path::PathBuf;

use velo_core::{
    compare_with_llama_csv, load_llama_csv, run_benchmark, BenchmarkConfig, BenchmarkFormat,
    BenchmarkMode,
};
use velo_core::metal::Quantization;
use velo_core::{EngineConfig, MemoryRuntimeConfig};

fn main() -> Result<(), Box<dyn Error>> {
    let args = CliArgs::parse()?;
    let engine_config = args.engine_config();
    let mut rows = Vec::new();

    for mode in args.modes() {
        let config = args.benchmark_config(mode);
        let mut report = run_benchmark(&engine_config, &config)?;
        rows.append(&mut report.rows);
    }

    let mut report = velo_core::BenchmarkReport { rows };

    if let Some(path) = &args.llama_csv {
        let csv = load_llama_csv(path)?;
        compare_with_llama_csv(&mut report.rows, &csv)
            .map_err(|error| format!("failed to compare with llama.cpp CSV: {error}"))?;
    }

    match args.format {
        BenchmarkFormat::Markdown => print!("{}", report.to_markdown()),
        BenchmarkFormat::Csv => print!("{}", report.to_csv()),
        BenchmarkFormat::Json => print!("{}", report.to_json()),
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct CliArgs {
    mode: String,
    prompt_len: usize,
    gen_len: usize,
    cached_depth: usize,
    repetitions: usize,
    warmups: usize,
    draft_window: usize,
    bytes_per_token: usize,
    page_tokens: usize,
    total_pages: usize,
    model_name: String,
    backend_name: String,
    quantization: Quantization,
    format: BenchmarkFormat,
    llama_csv: Option<PathBuf>,
}

impl CliArgs {
    fn parse() -> Result<Self, String> {
        let mut mode = "prompt-plus-generation".to_string();
        let mut prompt_len = 128usize;
        let mut gen_len = 128usize;
        let mut cached_depth = 0usize;
        let mut repetitions = 5usize;
        let mut warmups = 1usize;
        let mut draft_window = 8usize;
        let mut bytes_per_token = 128usize;
        let mut page_tokens = 16usize;
        let mut total_pages = 4096usize;
        let mut model_name = "velo-core/mock".to_string();
        let mut backend_name = "cpu/mock".to_string();
        let mut quantization = Quantization::Q4_0;
        let mut format = BenchmarkFormat::Markdown;
        let mut llama_csv = None;

        let mut args = env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--mode" => mode = take_value(&mut args, "--mode")?,
                "--prompt-len" => prompt_len = parse_usize(&mut args, "--prompt-len")?,
                "--gen-len" => gen_len = parse_usize(&mut args, "--gen-len")?,
                "--cached-depth" => cached_depth = parse_usize(&mut args, "--cached-depth")?,
                "--repetitions" => repetitions = parse_usize(&mut args, "--repetitions")?,
                "--warmups" => warmups = parse_usize(&mut args, "--warmups")?,
                "--draft-window" => draft_window = parse_usize(&mut args, "--draft-window")?,
                "--bytes-per-token" => bytes_per_token = parse_usize(&mut args, "--bytes-per-token")?,
                "--page-tokens" => page_tokens = parse_usize(&mut args, "--page-tokens")?,
                "--total-pages" => total_pages = parse_usize(&mut args, "--total-pages")?,
                "--model-name" => model_name = take_value(&mut args, "--model-name")?,
                "--backend-name" => backend_name = take_value(&mut args, "--backend-name")?,
                "--format" => {
                    let value = take_value(&mut args, "--format")?;
                    format = match value.as_str() {
                        "md" | "markdown" => BenchmarkFormat::Markdown,
                        "csv" => BenchmarkFormat::Csv,
                        "json" => BenchmarkFormat::Json,
                        _ => return Err(format!("unknown format: {value}")),
                    };
                }
                "--llama-csv" => {
                    llama_csv = Some(PathBuf::from(take_value(&mut args, "--llama-csv")?));
                }
                "--quantization" => {
                    let value = take_value(&mut args, "--quantization")?;
                    quantization = match value.as_str() {
                        "q4_0" | "q4" => Quantization::Q4_0,
                        "q4k" | "q4_k" | "q4-k" => Quantization::Q4K,
                        _ => return Err(format!("unknown quantization: {value}")),
                    };
                }
                "--help" | "-h" => return Err(Self::help()),
                other => return Err(format!("unrecognized argument: {other}")),
            }
        }

        Ok(Self {
            mode,
            prompt_len,
            gen_len,
            cached_depth,
            repetitions,
            warmups,
            draft_window,
            bytes_per_token,
            page_tokens,
            total_pages,
            model_name,
            backend_name,
            quantization,
            format,
            llama_csv,
        })
    }

    fn modes(&self) -> Vec<BenchmarkMode> {
        match self.mode.as_str() {
            "all" => vec![
                BenchmarkMode::PromptProcessing,
                BenchmarkMode::Generation,
                BenchmarkMode::PromptPlusGeneration,
            ],
            "prompt-processing" | "pp" => vec![BenchmarkMode::PromptProcessing],
            "generation" | "tg" => vec![BenchmarkMode::Generation],
            "prompt-plus-generation" | "pg" => vec![BenchmarkMode::PromptPlusGeneration],
            other => {
                eprintln!("unknown mode '{other}', defaulting to prompt-plus-generation");
                vec![BenchmarkMode::PromptPlusGeneration]
            }
        }
    }

    fn benchmark_config(&self, mode: BenchmarkMode) -> BenchmarkConfig {
        let (prompt_len, gen_len) = match mode {
            BenchmarkMode::PromptProcessing => (self.prompt_len, 0),
            BenchmarkMode::Generation => (0, self.gen_len),
            BenchmarkMode::PromptPlusGeneration => (self.prompt_len, self.gen_len),
        };

        BenchmarkConfig {
            mode,
            prompt_len,
            gen_len,
            cached_depth: self.cached_depth.min(prompt_len),
            repetitions: self.repetitions,
            warmups: self.warmups,
            draft_window: self.draft_window,
            bytes_per_token: self.bytes_per_token,
            page_tokens: self.page_tokens,
            total_pages: self.total_pages,
            quantization: self.quantization,
            model_name: self.model_name.clone(),
            backend_name: self.backend_name.clone(),
        }
    }

    fn engine_config(&self) -> EngineConfig {
        EngineConfig {
            draft_window: self.draft_window,
            memory: MemoryRuntimeConfig::cpu(self.bytes_per_token, self.page_tokens, self.total_pages, 32, 32),
        }
    }

    fn help() -> String {
        [
            "velo-bench",
            "",
            "Usage:",
            "  velo-bench [--mode prompt-processing|generation|prompt-plus-generation|all] [options]",
            "",
            "Options:",
            "  --prompt-len N",
            "  --gen-len N",
            "  --cached-depth N",
            "  --repetitions N",
            "  --warmups N",
            "  --draft-window N",
            "  --bytes-per-token N",
            "  --page-tokens N",
            "  --total-pages N",
            "  --format md|csv|json",
            "  --llama-csv PATH",
            "  --model-name NAME",
            "  --backend-name NAME",
            "  --quantization fp16|int8|int4",
        ]
        .join("\n")
    }
}

fn take_value<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_usize<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<usize, String>
where
    I: Iterator<Item = String>,
{
    let value = take_value(args, flag)?;
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid value for {flag}: {error}"))
}
