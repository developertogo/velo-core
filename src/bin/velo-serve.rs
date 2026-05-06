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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

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
    };

    let mut backend_config = MetalBackendConfig::default();
    backend_config.max_context_tokens = meta.n_ctx;
    backend_config.kv_bytes_per_token = 4;
    backend_config.paged_block_size = 16;
    backend_config.quantization = meta.quantization;

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
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, String> {
    // 1. Apply Chat Template (Flexible)
    let messages_json = serde_json::to_value(&req.messages).map_err(|e| e.to_string())?;
    let messages_slice = messages_json.as_array().ok_or("Failed to convert messages to array")?;
    let prompt = state.tokenizer.apply_chat_template(messages_slice, true)?;

    let token_ids = state.tokenizer.encode(&prompt);
    let (mut token_rx, _done_rx) = state.scheduler.submit(token_ids, req.max_tokens);

    let tokenizer = state.tokenizer.clone();
   
    let stream = async_stream::stream! {
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
            };
            yield Ok(Event::default().data(serde_json::to_string(&resp).unwrap()));
        }
       
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
        };
        yield Ok(Event::default().data(serde_json::to_string(&final_resp).unwrap()));
    };

    Ok(Sse::new(stream))
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
}
