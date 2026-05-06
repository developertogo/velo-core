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

#[derive(Debug, Deserialize)]
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
    // 1. Apply Llama 3 Template (Naive)
    let mut prompt = String::new();
    prompt.push_str("<|begin_of_text|>");
    for msg in req.messages {
        prompt.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", msg.role, msg.content));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let token_ids = state.tokenizer.encode(&prompt);
    let (mut token_rx, _done_rx) = state.scheduler.submit(token_ids, req.max_tokens);

    let tokenizer = state.tokenizer.clone();
   
    let stream = async_stream::stream! {
        while let Some(token) = token_rx.recv().await {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_template() {
        let req = ChatCompletionRequest {
            messages: vec![
                ChatMessage { role: "system".into(), content: "You are a bot".into() },
                ChatMessage { role: "user".into(), content: "Hello".into() },
            ],
            max_tokens: 10,
            stream: true,
        };

        let mut prompt = String::new();
        prompt.push_str("<|begin_of_text|>");
        for msg in req.messages {
            prompt.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", msg.role, msg.content));
        }
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

        assert!(prompt.contains("system"));
        assert!(prompt.contains("user"));
        assert!(prompt.contains("You are a bot"));
        assert!(prompt.contains("Hello"));
        assert!(prompt.contains("<|begin_of_text|>"));
    }
}
