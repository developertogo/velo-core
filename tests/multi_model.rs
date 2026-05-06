use std::path::PathBuf;
use std::time::Duration;
use velo_core::{VeloEngine, EngineConfig, MemoryRuntimeConfig, VeloScheduler, MockBackend, GreedyDraftModel, GreedyTargetModel};

#[tokio::test]
async fn test_scheduler_prefetch_and_switch() {
    // 1. Setup engine and scheduler
    let engine_config = EngineConfig {
        memory: MemoryRuntimeConfig::cpu(128, 16, 32, 32, 4),
        draft_window: 2,
        ..Default::default()
    };
    let engine = VeloEngine::new(engine_config).unwrap();
    
    let backend = MockBackend::new(vec![1, 2, 3, 4, 5]);
    let draft = GreedyDraftModel::new(backend.clone());
    let target = GreedyTargetModel::new(backend);
    
    let scheduler = VeloScheduler::start(engine, draft, target);
    
    // 2. Test prefetch command (mocked for now as we don't have a real GGUF)
    // This will print to stdout and call pool.prefetch which spawns a thread.
    // Since the path doesn't exist, it will eventually log an error in the background.
    scheduler.prefetch("new-model".to_string(), PathBuf::from("non-existent.gguf"));
    
    // 3. Test switch models (global)
    // This will attempt to switch to "new-model" which isn't in the pool yet.
    // It should fail gracefully (logging error) and keep using the old models.
    scheduler.switch_models("new-model".to_string(), Some("new-model".to_string()));
    
    // 4. Verify scheduler still works
    let (mut token_rx, done_rx) = scheduler.submit(vec![1], 2);
    let mut tokens = Vec::new();
    while let Some(Ok(token)) = token_rx.recv().await {
        tokens.push(token);
        if tokens.len() == 2 { break; }
    }
    assert_eq!(tokens.len(), 2);
    let _ = done_rx.await.unwrap().unwrap();
    
    // Give some time for background tasks
    tokio::time::sleep(Duration::from_millis(100)).await;
}
