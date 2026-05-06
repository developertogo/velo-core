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

/// A request sent to the scheduler.
pub struct SchedulerRequest {
    pub prompt: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub token_tx: mpsc::UnboundedSender<Result<TokenId, EngineError>>,
    pub done_tx: oneshot::Sender<Result<(), EngineError>>,
    /// Optional model name for this request (if pool-based).
    pub model: Option<String>,
}

/// Commands for administrative tasks in the scheduler.
pub enum SchedulerCommand {
    /// Load a model from disk into the pool in the background.
    Prefetch { name: String, path: std::path::PathBuf },
    /// Set the default models used for new requests.
    SwitchModels { target: String, draft: Option<String> },
}

/// A high-level asynchronous scheduler for the Velo Engine.
///
/// It handles request admission, continuous batching, and provides
/// an async interface for submitting requests.
pub struct VeloScheduler {
    request_tx: mpsc::UnboundedSender<SchedulerRequest>,
    command_tx: mpsc::UnboundedSender<SchedulerCommand>,
    metrics: Arc<SchedulerMetrics>,
}

/// Real-time metrics for the Velo Scheduler.
#[derive(Default)]
pub struct SchedulerMetrics {
    pub total_tokens_generated: AtomicU64,
    pub total_requests_completed: AtomicU64,
    pub active_slots: AtomicUsize,
    pub total_ttft_ms: AtomicU64,
    pub requests_with_ttft: AtomicU64,
    pub scheduler_start_time: Option<Instant>,
}

impl VeloScheduler {
    /// Starts a new scheduler worker in the background.
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

        tokio::spawn(async move {
            let mut active: Vec<ActiveRequestInternal> = Vec::new();
            let mut pending = VecDeque::new();
            
            // Initialize model pool with hardware handles from the engine
            let pool = crate::model_pool::ModelPool::new(
                engine.runtime().metal_handles().unwrap_or(MetalRuntimeHandles {
                    device: None, command_queue: None, library: None
                })
            );

            loop {
                // 1. Process commands
                while let Ok(cmd) = command_rx.try_recv() {
                    match cmd {
                        SchedulerCommand::Prefetch { name, path } => {
                            println!("Prefetching model {} from {:?}", name, path);
                            pool.prefetch(name, path);
                        }
                        SchedulerCommand::SwitchModels { target, draft } => {
                            println!("Switching models: target={}, draft={:?}", target, draft);
                            if let Err(e) = target_model.switch_model(&target, &pool) {
                                eprintln!("Failed to switch target model: {:?}", e);
                            }
                            if let Some(draft_name) = draft {
                                if let Err(e) = draft_model.switch_model(&draft_name, &pool) {
                                    eprintln!("Failed to switch draft model: {:?}", e);
                                }
                            }
                        }
                    }
                }

                // 2. Collect new requests
                while let Ok(req) = request_rx.try_recv() {
                    pending.push_back(req);
                }

                // 2. Admit pending requests into free slots
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

                if active.is_empty() && pending.is_empty() {
                    // Wait for new requests if nothing is active
                    if let Some(req) = request_rx.recv().await {
                        pending.push_back(req);
                        continue;
                    } else {
                        break; // Channel closed
                    }
                }

                // 3. Run one step of speculative decoding for all active requests
                if !active.is_empty() {
                    if let Err(e) = Self::step(&mut engine, &mut draft_model, &mut target_model, &mut active, &worker_metrics).await {
                        eprintln!("Scheduler step failed: {:?}", e);
                    }
                }

                // 4. Remove finished/cancelled requests
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

                // Yield to allow other tasks to run
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
        let draft_reqs: Vec<(&[TokenId], usize)> = active
            .iter()
            .map(|req| {
                let remaining = req.max_new_tokens - req.generated_count;
                let window = engine.decoder().draft_window().min(remaining);
                (req.session.context(), window)
            })
            .collect();

        let draft_results = draft_model.draft_batch(&draft_reqs)?;

        // 2. Verify
        let mut drafted_tokens_storage = Vec::with_capacity(active.len());
        for i in 0..active.len() {
            let drafted: Vec<TokenId> = draft_results[i].iter().map(|p| p.token).collect();
            drafted_tokens_storage.push(drafted);
        }

        let verify_reqs: Vec<(&[TokenId], &[TokenId])> = active
            .iter()
            .enumerate()
            .map(|(i, req)| {
                (req.session.context(), drafted_tokens_storage[i].as_slice())
            })
            .collect();

        let verify_results = target_model.verify_batch(&verify_reqs)?;

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
       
        let prompt = vec![1, 2, 3];
        let (mut token_rx, done_rx) = scheduler.submit(prompt, 5);
       
        // Wait for some tokens
        let mut tokens = Vec::new();
        while tokens.len() < 5 {
            if let Some(Ok(token)) = token_rx.recv().await {
                tokens.push(token);
            } else {
                break;
            }
        }
       
        assert_eq!(tokens.len(), 5);
        let _ = done_rx.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_scheduler_concurrency() {
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
       
        let (mut rx1, done1) = scheduler.submit(vec![1, 1], 2);
        let (mut rx2, done2) = scheduler.submit(vec![2, 2], 2);
       
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
        drop(rx); // Immediate cancellation

        // Wait a bit for scheduler to process
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(scheduler.metrics().active_slots.load(Ordering::Relaxed), 0);
    }
}
