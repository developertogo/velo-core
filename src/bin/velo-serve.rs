#![cfg(feature = "serve")]
use std::net::SocketAddr;
use std::sync::Arc;
use axum::{
    extract::State,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::path::Path;
use velo_core::paged_attention::KvCacheType;
use velo_core::{
    load_gguf, EngineConfig, MetalBackend, MetalBackendConfig, MetalMemoryRuntime,
    VeloEngine, VeloScheduler, tokenizer::Tokenizer,
    GreedyDraftModel, GreedyTargetModel, MetalRuntimeConfig, MemoryRuntimeConfig,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Number of slots (concurrency)
    #[arg(short, long, default_value_t = 4)]
    slots: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    #[allow(dead_code)]
    stream: bool,
}

fn default_max_tokens() -> usize { 128 }

#[derive(Debug, Serialize)]
struct ChatStreamResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatStreamChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Debug, Serialize)]
struct ChatUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    energy_uj: Option<u128>,
    hardware_utilization: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ChatStreamChoice {
    index: usize,
    delta: ChatDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatDelta {
    content: Option<String>,
}

struct AppState {
    scheduler: VeloScheduler,
    tokenizer: Tokenizer,
}

#[cfg(all(target_os = "linux", feature = "ebpf"))]
fn setup_ebpf_observer() -> anyhow::Result<()> {
    println!("Initializing eBPF latency observer...");
    // In a real deployment, this would load the compiled BPF bytecode
    // and attach to tracepoints like sched:sched_switch and syscalls:sys_enter_accept
    Ok(())
}

#[cfg(not(all(target_os = "linux", feature = "ebpf")))]
fn setup_ebpf_observer() -> anyhow::Result<()> {
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    setup_ebpf_observer()?;

    println!("Loading model from {}...", args.model);
    let weights = load_gguf(Path::new(&args.model))?;
    let meta = weights.meta.clone();
   
    // We need the raw GgufFile for the tokenizer
    let mut file = std::fs::File::open(&args.model)?;
    let gguf = velo_core::GgufFile::read(&mut file)?;
    let tokenizer = Tokenizer::from_gguf(&gguf);

    let memory_config = MemoryRuntimeConfig {
        bytes_per_token: 4, // f32
        paged_block_size: 16,
        paged_total_pages: 1024,
        n_layer: meta.n_layer,
        unified_memory: true,
        max_slots: args.slots,
        kv_type: KvCacheType::Fp32,
    };

    let runtime_config = MetalRuntimeConfig {
        model_name: "velo-llama".into(),
        memory: memory_config,
        quantization: meta.quantization,
        tensor_parallel_degree: 1,
    };

    let mut backend_config = MetalBackendConfig::default();
    backend_config.max_context_tokens = meta.n_ctx;
    backend_config.kv_bytes_per_token = 4;
    backend_config.paged_block_size = 16;
    backend_config.quantization = meta.quantization;
    backend_config.tensor_parallel_degree = 1;

    let mut backend = MetalBackend::new(backend_config)?;
    let engine_config = EngineConfig {
        memory: memory_config,
        draft_window: 4,
        kv_type: KvCacheType::Fp32,
    };
   
    let runtime = MetalMemoryRuntime::new(runtime_config)?;
    backend.wire(weights, &runtime)?;
   
    let engine = VeloEngine::with_runtime(engine_config, runtime)?;
   
    let draft_model = GreedyDraftModel::new(backend.clone());
    let target_model = GreedyTargetModel::new(backend);

    let scheduler = VeloScheduler::start(engine, draft_model, target_model);

    let state = Arc::new(AppState {
        scheduler,
        tokenizer,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/metrics", get(metrics_handler))
        .route("/health", get(|| async { "OK" }))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    println!("Velo-Core serving on http://{}", addr);
   
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Handler for the OpenAI-compatible chat completions endpoint.
///
/// It applies a naive Llama 3 chat template to the messages and
/// streams the generated tokens using Server-Sent Events (SSE).
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<(axum::http::HeaderMap, Sse<impl Stream<Item = Result<Event, Infallible>>>), String> {
    // 1. Apply Chat Template (Flexible)
    let messages_json = serde_json::to_value(&req.messages).map_err(|e| e.to_string())?;
    let messages_slice = messages_json.as_array().ok_or("Failed to convert messages to array")?;
    let prompt = state.tokenizer.apply_chat_template(messages_slice, true)?;

    let token_ids = state.tokenizer.encode(&prompt);
    let prompt_len = token_ids.len();
    let (mut token_rx, _done_rx) = state.scheduler.submit(token_ids, req.max_tokens);

    let tokenizer = state.tokenizer.clone();
    let specs = velo_core::HardwareSpecs::detect();
    
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("X-Velo-Hardware-Peak-GBs", specs.peak_bw_gb_s.to_string().parse().unwrap());
    headers.insert("X-Velo-Hardware-Peak-TFLOPS", specs.peak_tflops.to_string().parse().unwrap());
    
    let stream = async_stream::stream! {
        let prompt_tokens = prompt_len;
        let mut completion_tokens = 0;
        let started = std::time::Instant::now();
        while let Some(res) = token_rx.recv().await {
            let token = match res {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Scheduler error: {:?}", e);
                    break;
                }
            };
            let text = tokenizer.decode(&[token]);
            let resp = ChatStreamResponse {
                id: "velo-123".into(),
                object: "chat.completion.chunk".into(),
                created: 123456789,
                model: "velo-llama".into(),
                choices: vec![ChatStreamChoice {
                    index: 0,
                    delta: ChatDelta { content: Some(text) },
                    finish_reason: None,
                }],
                usage: None,
            };
            completion_tokens += 1;
            yield Ok(Event::default().data(serde_json::to_string(&resp).unwrap()));
        }
       
        let elapsed_ns = started.elapsed().as_nanos();
        let total_tokens = prompt_tokens + completion_tokens;
        
        // Naive utilization estimate for telemetry
        let util = if elapsed_ns > 0 {
             // Weights (8B Q4) + KV etc. Roughly 4.5 GB read per token
             let bytes_per_token = 4.5e9;
             let total_bytes = completion_tokens as f64 * bytes_per_token;
             let gb_s = (total_bytes / 1e9) / (elapsed_ns as f64 / 1e9);
             Some(gb_s / specs.peak_bw_gb_s)
        } else {
             None
        };

        let final_resp = ChatStreamResponse {
            id: "velo-123".into(),
            object: "chat.completion.chunk".into(),
            created: 123456789,
            model: "velo-llama".into(),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatDelta { content: None },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(ChatUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                energy_uj: None, // Hard to measure per-request in scheduler
                hardware_utilization: util,
            }),
        };
        yield Ok(Event::default().data(serde_json::to_string(&final_resp).unwrap()));
    };

    Ok((headers, Sse::new(stream)))
}

async fn list_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "velo-llama",
                "object": "model",
                "created": 123456789,
                "owned_by": "velo"
            }
        ]
    }))
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> String {
    let m = state.scheduler.metrics();
    let total_tokens = m.total_tokens_generated.load(std::sync::atomic::Ordering::Relaxed);
    let total_reqs = m.total_requests_completed.load(std::sync::atomic::Ordering::Relaxed);
    let active_slots = m.active_slots.load(std::sync::atomic::Ordering::Relaxed);
    let total_ttft = m.total_ttft_ms.load(std::sync::atomic::Ordering::Relaxed);
    let reqs_with_ttft = m.requests_with_ttft.load(std::sync::atomic::Ordering::Relaxed);
    
    let avg_ttft = if reqs_with_ttft > 0 {
        total_ttft as f64 / reqs_with_ttft as f64
    } else {
        0.0
    };

    let uptime = if let Some(start) = m.scheduler_start_time {
        start.elapsed().as_secs()
    } else {
        0
    };

    let tps = if uptime > 0 {
        total_tokens as f64 / uptime as f64
    } else {
        0.0
    };

    format!(
        "# HELP velo_tokens_total Total tokens generated\n\
         # TYPE velo_tokens_total counter\n\
         velo_tokens_total {}\n\n\
         # HELP velo_requests_total Total requests completed\n\
         # TYPE velo_requests_total counter\n\
         velo_requests_total {}\n\n\
         # HELP velo_active_slots Current active slots\n\
         # TYPE velo_active_slots gauge\n\
         velo_active_slots {}\n\n\
         # HELP velo_ttft_ms_avg Average Time To First Token in ms\n\
         # TYPE velo_ttft_ms_avg gauge\n\
         velo_ttft_ms_avg {:.2}\n\n\
         # HELP velo_tokens_per_second Average tokens per second since start\n\
         # TYPE velo_tokens_per_second gauge\n\
         velo_tokens_per_second {:.2}\n\n\
         # HELP velo_uptime_seconds Seconds since scheduler start\n\
         # TYPE velo_uptime_seconds counter\n\
         velo_uptime_seconds {}\n",
        total_tokens, total_reqs, active_slots, avg_ttft, tps, uptime
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_template() {
        let mut tokenizer = Tokenizer::from_gguf(&velo_core::gguf::GgufFile {
            version: 3,
            metadata: std::collections::HashMap::new(),
            tensors: std::collections::HashMap::new(),
            data_offset: 0,
        });
        // Set a custom template for testing
        tokenizer.chat_template = Some("{{ messages[0].content }}".to_string());

        let messages = vec![
            serde_json::json!({ "role": "user", "content": "Hello" }),
        ];

        let prompt = tokenizer.apply_chat_template(&messages, true).unwrap();
        assert_eq!(prompt, "Hello");
    }

    #[test]
    fn test_default_max_tokens() {
        assert_eq!(default_max_tokens(), 128);
    }

    #[test]
    fn test_chat_stream_response_serialization() {
        let resp = ChatStreamResponse {
            id: "id-1".into(),
            object: "chat.completion.chunk".into(),
            created: 0,
            model: "test".into(),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatDelta { content: Some("hello".into()) },
                finish_reason: None,
            }],
            usage: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("hello"));
        assert!(json.contains("chat.completion.chunk"));
    }

    #[test]
    fn test_chat_usage_serialization() {
        let usage = ChatUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            energy_uj: Some(1000),
            hardware_utilization: Some(0.75),
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("\"prompt_tokens\":10"));
        assert!(json.contains("0.75"));
    }

    #[test]
    fn test_chat_message_roundtrip() {
        let msg = ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "user");
        assert_eq!(back.content, "hi");
    }

    #[test]
    fn test_chat_completion_request_defaults() {
        let json = r#"{"messages":[]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 128);
        assert!(!req.stream);
    }
}
