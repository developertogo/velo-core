use velo_core::metal::model::LlamaMetalModel;
use velo_core::metal::runtime::MetalMemoryRuntime;
use velo_core::metal::config::MetalRuntimeConfig;
use velo_core::model_loader::{WeightStore, ModelMeta};
use velo_core::metal::Quantization;
use velo_core::paged_attention::KvCacheType;
use velo_core::slot_manager::SlotId;
use objc2_metal::{MTLBuffer, MTLCommandQueue};

fn test_config() -> MetalRuntimeConfig {
    let mut config = MetalRuntimeConfig::default();
    config.memory.bytes_per_token = 128;
    config.memory.paged_block_size = 16;
    config.memory.paged_total_pages = 100;
    config
}

#[tokio::test]
async fn test_gpu_argmax_kernel() {
    let runtime_res = MetalMemoryRuntime::new(test_config());
    if let Err(e) = runtime_res {
        if format!("{:?}", e).contains("No Metal device found") {
            eprintln!("Skipping test: No Metal device found");
            return;
        }
        panic!("Failed to create runtime: {:?}", e);
    }
    let runtime = runtime_res.unwrap();
    let n_vocab = 32000;
    
    let meta = ModelMeta {
        arch: "llama".into(),
        n_vocab,
        n_embd: 128,
        n_layer: 1,
        n_head: 4,
        n_head_kv: 1,
        n_ctx: 1024,
        n_ff: 512,
        head_dim: 32,
        rope_freq_base: 10000.0,
        norm_eps: 1e-5,
        quantization: Quantization::F32,
    };
    
    let handles = runtime.context().handles.clone();
    let device = handles.device.as_ref().unwrap().clone();
    let queue = handles.command_queue.as_ref().unwrap().clone();
    let library = handles.library.as_ref().unwrap().clone();
    
    let mut model = LlamaMetalModel::new(meta, device, queue, library);
    
    let mut logits = vec![-1.0f32; n_vocab];
    let expected_token = 1234;
    logits[expected_token] = 100.0;
    
    let logits_buf = model.get_scratch("test_logits", n_vocab * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(logits.as_ptr(), logits_buf.contents().as_ptr() as *mut f32, n_vocab);
    }
    
    let command_buffer = model.queue.commandBuffer().unwrap();
    let result_token = model.sample_argmax(&command_buffer, &logits_buf).unwrap();
    
    assert_eq!(result_token, expected_token as u32);
}

#[tokio::test]
async fn test_flash_attention_parity_smoke() {
    let runtime_res = MetalMemoryRuntime::new(test_config());
    if let Err(e) = runtime_res {
        if format!("{:?}", e).contains("No Metal device found") {
            eprintln!("Skipping test: No Metal device found");
            return;
        }
        panic!("Failed to create runtime: {:?}", e);
    }
    let runtime = runtime_res.unwrap();
    let n_vocab = 1000;
    let n_embd = 128;
    
    let weights = WeightStore::dummy_llama(n_vocab, n_embd, 1);
    let mut model = LlamaMetalModel::new(
        weights.meta.clone(),
        runtime.context().handles.device.as_ref().unwrap().clone(),
        runtime.context().handles.command_queue.as_ref().unwrap().clone(),
        runtime.context().handles.library.as_ref().unwrap().clone()
    );
    model.upload_weights(&weights).unwrap();
    
    let slot_id = SlotId(0);
    let pos = 0;
    let token = 1;
    
    let k_pool = runtime.store.0.lock().unwrap().k_pool().clone();
    let v_pool = runtime.store.0.lock().unwrap().v_pool().clone();
    let total_pages = runtime.allocator.0.lock().unwrap().config().total_pages;

    let res = model.run(
        token,
        pos,
        slot_id,
        &runtime.slot_mapping,
        &k_pool,
        &v_pool,
        total_pages,
        16,
        KvCacheType::Fp32,
    ).unwrap();
    
    assert_eq!(res.len(), n_vocab);
    assert!(!res.iter().any(|&x| x.is_nan()));
}
