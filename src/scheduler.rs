use std::collections::VecDeque;
use tokio::sync::{mpsc, oneshot};
use crate::radix_cache::TokenId;
use crate::slot_manager::SlotId;
use crate::engine::{VeloEngine, EngineError};
use crate::speculative::{DraftModel, TargetModel, SpeculativeError};
use crate::runtime::MemoryRuntime;

/// A request sent to the scheduler.
pub struct SchedulerRequest {
    pub prompt: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub token_tx: mpsc::UnboundedSender<TokenId>,
    pub done_tx: oneshot::Sender<()>,
}

/// A high-level asynchronous scheduler for the Velo Engine.
///
/// It handles request admission, continuous batching, and provides
/// an async interface for submitting requests.
pub struct VeloScheduler {
    request_tx: mpsc::UnboundedSender<SchedulerRequest>,
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

        tokio::spawn(async move {
            let mut active: Vec<ActiveRequestInternal> = Vec::new();
            let mut pending = VecDeque::new();

            loop {
                // 1. Collect new requests from the channel
                while let Ok(req) = request_rx.try_recv() {
                    pending.push_back(req);
                }

                // 2. Admit pending requests into free slots
                while active.len() < engine.slot_pool_capacity() && !pending.is_empty() {
                    if let Some(req) = pending.pop_front() {
                        match Self::admit_request(&mut engine, &mut draft_model, &mut target_model, req) {
                            Ok(active_req) => active.push(active_req),
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
                    if let Err(e) = Self::step(&mut engine, &mut draft_model, &mut target_model, &mut active).await {
                        eprintln!("Scheduler step failed: {:?}", e);
                    }
                }

                // 4. Remove finished requests
                let mut i = 0;
                while i < active.len() {
                    if active[i].generated_count >= active[i].max_new_tokens || active[i].finished {
                        let req = active.remove(i);
                        engine.release_slot(req.slot_id);
                        let _ = req.done_tx.send(());
                    } else {
                        i += 1;
                    }
                }

                // Yield to allow other tasks to run
                tokio::task::yield_now().await;
            }
        });

        Self { request_tx }
    }

    /// Submits a request to the scheduler and returns a receiver for tokens.
    pub fn submit(&self, prompt: Vec<TokenId>, max_new_tokens: usize) -> (mpsc::UnboundedReceiver<TokenId>, oneshot::Receiver<()>) {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        let (done_tx, done_rx) = oneshot::channel();
       
        let req = SchedulerRequest {
            prompt,
            max_new_tokens,
            token_tx,
            done_tx,
        };

        let _ = self.request_tx.send(req);
        (token_rx, done_rx)
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
        })
    }

    async fn step<R, D, T>(
        engine: &mut VeloEngine<R>,
        draft_model: &mut D,
        target_model: &mut T,
        active: &mut [ActiveRequestInternal],
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
                let _ = req.token_tx.send(token);
                req.generated_count += 1;
            }

            if req.session.has_pending_rejection() {
                if let Some(token) = req.session.take_rejected_token() {
                    let _ = req.token_tx.send(token);
                    req.generated_count += 1;
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
    token_tx: mpsc::UnboundedSender<TokenId>,
    done_tx: oneshot::Sender<()>,
    generated_count: usize,
    finished: bool,
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
            if let Some(token) = token_rx.recv().await {
                tokens.push(token);
            } else {
                break;
            }
        }
       
        assert_eq!(tokens.len(), 5);
        let _ = done_rx.await;
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
        };
        let engine = VeloEngine::new(engine_config).unwrap();
       
        let draft_model = GreedyDraftModel::new(backend.clone());
        let target_model = GreedyTargetModel::new(backend);
       
        let scheduler = VeloScheduler::start(engine, draft_model, target_model);
       
        let (mut rx1, done1) = scheduler.submit(vec![1, 1], 2);
        let (mut rx2, done2) = scheduler.submit(vec![2, 2], 2);
       
        let h1 = tokio::spawn(async move {
            let mut count = 0;
            while rx1.recv().await.is_some() { count += 1; if count == 2 { break; } }
            done1.await.unwrap();
        });
       
        let h2 = tokio::spawn(async move {
            let mut count = 0;
            while rx2.recv().await.is_some() { count += 1; if count == 2 { break; } }
            done2.await.unwrap();
        });
       
        h1.await.unwrap();
        h2.await.unwrap();
    }
}
