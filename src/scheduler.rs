//! VeloScheduler: The Traffic Controller
//!
//! While `VeloEngine` handles the math and memory, the `VeloScheduler` handles the "crowd".
//! It manages a queue of people waiting to talk to the AI and decides when to let them in.
//!
//! ### Key Concepts for Beginners:
//! - **Continuous Batching**: Imagine a bus (the GPU) that never stops. As soon as one 
//!   passenger gets off (a request finishes), another passenger gets on (a new request starts),
//!   even if the other passengers are still in the middle of their journey.
//! - **Worker Loop**: A background thread that constantly runs the engine's "step" function.
//! - **Async Interface**: Allowing the rest of the application to "submit and wait" for tokens
//!   without blocking the whole system.

use std::collections::VecDeque;
use tokio::sync::{mpsc, oneshot};
use crate::radix_cache::TokenId;
use crate::slot_manager::SlotId;
use crate::engine::{VeloEngine, EngineError};
use crate::speculative::{DraftModel, TargetModel, SpeculativeError};
use crate::runtime::MemoryRuntime;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use crate::metal::runtime::MetalRuntimeHandles;

/// A single request sent to the scheduler by a user.
pub struct SchedulerRequest {
    /// The prompt tokens.
    pub prompt: Vec<TokenId>,
    /// How many tokens to generate.
    pub max_new_tokens: usize,
    /// A "pipe" (channel) to send generated tokens back to the user as they happen.
    pub token_tx: mpsc::UnboundedSender<Result<TokenId, EngineError>>,
    /// A notification channel to tell the user when we are completely done.
    pub done_tx: oneshot::Sender<Result<(), EngineError>>,
    /// Optional model name if we are using a pool of different models.
    pub model: Option<String>,
}

/// Commands for administrative tasks (like loading new models).
pub enum SchedulerCommand {
    /// Load a model from disk into the pool.
    Prefetch { name: String, path: std::path::PathBuf },
    /// Change which models are currently being used for inference.
    SwitchModels { target: String, draft: Option<String> },
}

/// A high-level asynchronous scheduler for the Velo Engine.
///
/// This is the main entry point for most applications. It runs a background
/// task that manages the engine and processes requests.
pub struct VeloScheduler {
    /// Channel for sending new requests to the background worker.
    request_tx: mpsc::UnboundedSender<SchedulerRequest>,
    /// Channel for sending commands (like switching models).
    command_tx: mpsc::UnboundedSender<SchedulerCommand>,
    /// Real-time statistics about throughput and latency.
    metrics: Arc<SchedulerMetrics>,
}

/// Real-time metrics for monitoring the scheduler's health.
#[derive(Default)]
pub struct SchedulerMetrics {
    pub total_tokens_generated: AtomicU64,
    pub total_requests_completed: AtomicU64,
    /// How many GPU slots are currently being used.
    pub active_slots: AtomicUsize,
    /// Total "Time To First Token" in milliseconds.
    pub total_ttft_ms: AtomicU64,
    pub requests_with_ttft: AtomicU64,
    pub scheduler_start_time: Option<Instant>,
}

impl VeloScheduler {
    /// Starts the scheduler worker loop in a background thread.
    /// This is where the actual "work" happens.
    pub fn start<R, D, T>(
        mut engine: VeloEngine<R>,
        mut draft_model: D,
        mut target_model: T,
    ) -> Self
    where
        R: MemoryRuntime + Send + 'static,
        D: DraftModel + Send + 'static,
        T: TargetModel + Send + 'static,
    {
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<SchedulerRequest>();
        let (command_tx, mut command_rx) = mpsc::unbounded_channel::<SchedulerCommand>();
        let metrics = Arc::new(SchedulerMetrics {
            scheduler_start_time: Some(Instant::now()),
            ..Default::default()
        });
        let worker_metrics = metrics.clone();

        // Spawn the background worker thread.
        tokio::spawn(async move {
            let mut active: Vec<ActiveRequestInternal> = Vec::new();
            let mut pending = VecDeque::new();
            
            // The ModelPool allows us to keep multiple models ready to go on the GPU.
            let pool = crate::model_pool::ModelPool::new(
                engine.runtime().metal_handles().unwrap_or(MetalRuntimeHandles {
                    device: None, command_queue: None, library: None, tp_degree: 1, tp_rank: 0
                })
            );

            loop {
                // 1. Check for administrative commands (like loading a new model).
                while let Ok(cmd) = command_rx.try_recv() {
                    match cmd {
                        SchedulerCommand::Prefetch { name, path } => {
                            pool.prefetch(name, path);
                        }
                        SchedulerCommand::SwitchModels { target, draft } => {
                            let _ = target_model.switch_model(&target, &pool);
                            if let Some(draft_name) = draft {
                                let _ = draft_model.switch_model(&draft_name, &pool);
                            }
                        }
                    }
                }

                // 2. Look for brand new requests from users.
                while let Ok(req) = request_rx.try_recv() {
                    pending.push_back(req);
                }

                // 3. ADMISSION: If we have free slots on the GPU, let waiting passengers (requests) on.
                while active.len() < engine.slot_pool_capacity() && !pending.is_empty() {
                    if let Some(req) = pending.pop_front() {
                        match Self::admit_request(&mut engine, &mut draft_model, &mut target_model, req) {
                            Ok(active_req) => {
                                active.push(active_req);
                                worker_metrics.active_slots.store(active.len(), Ordering::Relaxed);
                            }
                            Err(e) => {
                                eprintln!("Failed to admit request: {:?}", e);
                            }
                        }
                    }
                }

                // If nothing is happening, wait for a new request.
                if active.is_empty() && pending.is_empty() {
                    if let Some(req) = request_rx.recv().await {
                        pending.push_back(req);
                        continue;
                    } else {
                        break; // All channels closed, stop the scheduler.
                    }
                }

                // 4. STEP: Run one round of token generation for everyone currently "on the bus".
                if !active.is_empty() {
                    if let Err(e) = Self::step(&mut engine, &mut draft_model, &mut target_model, &mut active, &worker_metrics).await {
                        eprintln!("Scheduler step failed: {:?}", e);
                    }
                }

                // 5. CLEANUP: Remove requests that have finished their job or were cancelled.
                let mut i = 0;
                while i < active.len() {
                    let is_done = active[i].generated_count >= active[i].max_new_tokens || active[i].finished;
                    let is_cancelled = active[i].cancelled || active[i].token_tx.is_closed();
                    
                    if is_done || is_cancelled {
                        let req = active.remove(i);
                        engine.release_slot(req.slot_id);
                        worker_metrics.active_slots.store(active.len(), Ordering::Relaxed);
                        if is_done && !is_cancelled {
                            worker_metrics.total_requests_completed.fetch_add(1, Ordering::Relaxed);
                            let _ = req.done_tx.send(Ok(()));
                        }
                    } else {
                        i += 1;
                    }
                }

                // Allow other parts of the program to run briefly.
                tokio::task::yield_now().await;
            }
        });

        Self { request_tx, command_tx, metrics }
    }

    /// Returns a reference to the scheduler metrics.
    pub fn metrics(&self) -> Arc<SchedulerMetrics> {
        self.metrics.clone()
    }

    /// Submits a request to the scheduler and returns a receiver for tokens.
    pub fn submit(&self, prompt: Vec<TokenId>, max_new_tokens: usize) -> (mpsc::UnboundedReceiver<Result<TokenId, EngineError>>, oneshot::Receiver<Result<(), EngineError>>) {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        let (done_tx, done_rx) = oneshot::channel();
       
        let req = SchedulerRequest {
            prompt,
            max_new_tokens,
            token_tx,
            done_tx,
            model: None,
        };

        let _ = self.request_tx.send(req);
        (token_rx, done_rx)
    }

    /// Prefetches a model in the background.
    pub fn prefetch(&self, name: String, path: std::path::PathBuf) {
        let _ = self.command_tx.send(SchedulerCommand::Prefetch { name, path });
    }

    /// Switches the default models.
    pub fn switch_models(&self, target: String, draft: Option<String>) {
        let _ = self.command_tx.send(SchedulerCommand::SwitchModels { target, draft });
    }

    fn admit_request<R, D, T>(
        engine: &mut VeloEngine<R>,
        draft_model: &mut D,
        target_model: &mut T,
        req: SchedulerRequest,
    ) -> Result<ActiveRequestInternal, EngineError>
    where
        R: MemoryRuntime,
        D: DraftModel,
        T: TargetModel,
    {
        let slot_id = engine.allocate_slot().ok_or_else(|| {
            EngineError::Speculative(SpeculativeError::Model("No free slots".to_string()))
        })?;

        let prefill = engine.prefill(&req.prompt)?;
       
        // Bind slot and prefix cache
        draft_model.bind_slot(slot_id)?;
        target_model.bind_slot(slot_id)?;
        draft_model.bind_prefix_cache(&prefill.cached_prefix)?;
        target_model.bind_prefix_cache(&prefill.cached_prefix)?;

        let session = engine.decoder().begin(&req.prompt)?;

        Ok(ActiveRequestInternal {
            session,
            slot_id,
            max_new_tokens: req.max_new_tokens,
            token_tx: req.token_tx,
            done_tx: req.done_tx,
            generated_count: 0,
            finished: false,
            start_time: Instant::now(),
            first_token_sent: false,
            cancelled: false,
        })
    }

    async fn step<R, D, T>(
        engine: &mut VeloEngine<R>,
        draft_model: &mut D,
        target_model: &mut T,
        active: &mut [ActiveRequestInternal],
        metrics: &SchedulerMetrics,
    ) -> Result<(), EngineError>
    where
        R: MemoryRuntime,
        D: DraftModel,
        T: TargetModel,
    {
        // This is a simplified version of generate_batch logic, but running only ONE step.
        // 1. Draft
        let mut draft_reqs: Vec<(&[TokenId], usize, Option<&mut (dyn crate::constraints::CfgMatcher + '_)>)> = active
            .iter_mut()
            .map(|req| {
                let remaining = req.max_new_tokens - req.generated_count;
                let window = engine.decoder().draft_window().min(remaining);
                (req.session.context(), window, None)
            })
            .collect();

        let draft_results = draft_model.draft_batch(&mut draft_reqs)?;

        // 2. Verify
        let mut drafted_tokens_storage = Vec::with_capacity(active.len());
        for i in 0..active.len() {
            let drafted: Vec<TokenId> = draft_results[i].iter().map(|p| p.token).collect();
            drafted_tokens_storage.push(drafted);
        }

        let mut verify_reqs: Vec<(&[TokenId], &[TokenId], Option<&mut (dyn crate::constraints::CfgMatcher + '_)>)> = active
            .iter_mut()
            .enumerate()
            .map(|(i, req)| {
                (req.session.context(), drafted_tokens_storage[i].as_slice(), None)
            })
            .collect();

        let verify_results = target_model.verify_batch(&mut verify_reqs)?;

        // 3. Commit and Send
        for (i, req) in active.iter_mut().enumerate() {
            let accepted = req.session.commit(&draft_results[i], &verify_results[i])?;
            for token in accepted {
                if !req.first_token_sent {
                    let ttft = req.start_time.elapsed().as_millis() as u64;
                    metrics.total_ttft_ms.fetch_add(ttft, Ordering::Relaxed);
                    metrics.requests_with_ttft.fetch_add(1, Ordering::Relaxed);
                    req.first_token_sent = true;
                }
                if req.token_tx.send(Ok(token)).is_err() {
                    req.cancelled = true;
                    break;
                }
                req.generated_count += 1;
                metrics.total_tokens_generated.fetch_add(1, Ordering::Relaxed);
            }
            if req.cancelled { continue; }

            if req.session.has_pending_rejection() {
                if let Some(token) = req.session.take_rejected_token() {
                    if !req.first_token_sent {
                        let ttft = req.start_time.elapsed().as_millis() as u64;
                        metrics.total_ttft_ms.fetch_add(ttft, Ordering::Relaxed);
                        metrics.requests_with_ttft.fetch_add(1, Ordering::Relaxed);
                        req.first_token_sent = true;
                    }
                    if req.token_tx.send(Ok(token)).is_err() {
                        req.cancelled = true;
                    }
                    req.generated_count += 1;
                    metrics.total_tokens_generated.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        Ok(())
    }
}

struct ActiveRequestInternal {
    session: crate::speculative::SpeculativeSession,
    slot_id: SlotId,
    max_new_tokens: usize,
    token_tx: mpsc::UnboundedSender<Result<TokenId, EngineError>>,
    done_tx: oneshot::Sender<Result<(), EngineError>>,
    generated_count: usize,
    finished: bool,
    start_time: Instant,
    first_token_sent: bool,
    cancelled: bool,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_backend::MockBackend;
    use crate::backend::{GreedyDraftModel, GreedyTargetModel};

    #[tokio::test]
    async fn test_scheduler_basic_flow() {
        // TEST: Asynchronous "Ask and Receive"
        // In a real app, the user shouldn't have to wait for the whole paragraph
        // to be done. They want to see tokens appearing one by one.
        
        let script = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let backend = MockBackend::new(script);
       
        let engine_config = crate::engine::EngineConfig {
            memory: crate::runtime::MemoryRuntimeConfig {
                max_slots: 2,
                paged_total_pages: 10,
                paged_block_size: 4,
                ..Default::default()
            },
            draft_window: 2,
            kv_type: crate::paged_attention::KvCacheType::Fp32,
        };
        let engine = VeloEngine::new(engine_config).unwrap();
       
        let draft_model = GreedyDraftModel::new(backend.clone());
        let target_model = GreedyTargetModel::new(backend);
       
        // 1. Start the background worker loop.
        let scheduler = VeloScheduler::start(engine, draft_model, target_model);
       
        let prompt = vec![1, 2, 3];
        // 2. Submit a request. We get back a "pipe" (rx) for tokens.
        let (mut token_rx, done_rx) = scheduler.submit(prompt, 5);
       
        // 3. Collect tokens as they arrive from the background thread.
        let mut tokens = Vec::new();
        while tokens.len() < 5 {
            if let Some(Ok(token)) = token_rx.recv().await {
                tokens.push(token);
            } else {
                break;
            }
        }
       
        assert_eq!(tokens.len(), 5);
        // 4. Ensure the whole process finished successfully.
        let _ = done_rx.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_scheduler_concurrency() {
        // TEST: Multiple Users at Once
        // We simulate two people asking the AI different things simultaneously.
        // The scheduler must manage both "passengers" on the GPU bus.
        
        let script = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let backend = MockBackend::new(script);
       
        let engine_config = crate::engine::EngineConfig {
            memory: crate::runtime::MemoryRuntimeConfig {
                max_slots: 2,
                paged_total_pages: 10,
                paged_block_size: 4,
                ..Default::default()
            },
            draft_window: 2,
            kv_type: crate::paged_attention::KvCacheType::Fp32,
        };
        let engine = VeloEngine::new(engine_config).unwrap();
       
        let draft_model = GreedyDraftModel::new(backend.clone());
        let target_model = GreedyTargetModel::new(backend);
       
        let scheduler = VeloScheduler::start(engine, draft_model, target_model);
       
        // Submit two requests at the same time.
        let (mut rx1, done1) = scheduler.submit(vec![1, 1], 2);
        let (mut rx2, done2) = scheduler.submit(vec![2, 2], 2);
       
        // We use 'tokio::spawn' to simulate two independent users waiting for their answers.
        let h1 = tokio::spawn(async move {
            let mut count = 0;
            while let Some(Ok(_)) = rx1.recv().await { count += 1; if count == 2 { break; } }
            done1.await.unwrap().unwrap();
        });
       
        let h2 = tokio::spawn(async move {
            let mut count = 0;
            while let Some(Ok(_)) = rx2.recv().await { count += 1; if count == 2 { break; } }
            done2.await.unwrap().unwrap();
        });
       
        h1.await.unwrap();
        h2.await.unwrap();
    }

    #[tokio::test]
    async fn test_scheduler_cancellation() {
        // TEST: Cleaning up after a "Hang up"
        // If a user closes their tab (cancellation), we should immediately stop
        // using the GPU and free up their memory slot.
        
        let script = vec![1, 2, 3, 4, 5];
        let backend = MockBackend::new(script);
        let engine_config = crate::engine::EngineConfig {
            memory: crate::runtime::MemoryRuntimeConfig {
                max_slots: 1,
                paged_total_pages: 10,
                paged_block_size: 4,
                ..Default::default()
            },
            draft_window: 1,
            kv_type: crate::paged_attention::KvCacheType::Fp32,
        };
        let engine = VeloEngine::new(engine_config).unwrap();
        let scheduler = VeloScheduler::start(engine, GreedyDraftModel::new(backend.clone()), GreedyTargetModel::new(backend));

        let (rx, _done) = scheduler.submit(vec![1], 10);
        
        // Simulating the user "hanging up" by dropping the receiver.
        drop(rx); 

        // Give the background loop a few milliseconds to notice.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        
        // The slot should be empty now!
        assert_eq!(scheduler.metrics().active_slots.load(Ordering::Relaxed), 0);
    }
}
