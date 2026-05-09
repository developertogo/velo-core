use velo_core::model_loader::WeightStore;
use velo_core::llama_cpu::LlamaCpuModel;
use velo_core::metal::{MetalMemoryRuntime, MetalRuntimeConfig, LlamaMetalModel};
use velo_core::runtime::MemoryRuntimeConfig;
use velo_core::paged_attention::KvCacheType;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

#[test]
fn test_llama3_8b_parity() {
    // ── Setup ────────────────────────────────────────────────────────────────
    let n_vocab = 128;
    let n_embd = 4096;
    let n_layer = 2;
    let n_head = 32;
    let n_head_kv = 8;
    let head_dim = 128;
    let n_ff = 14336;
    
    let mut weights = WeightStore::dummy_llama(n_vocab, n_embd, n_layer);
    weights.meta.n_head = n_head;
    weights.meta.n_head_kv = n_head_kv;
    weights.meta.head_dim = head_dim;
    weights.meta.n_ff = n_ff;
    weights.meta.rope_freq_base = 500_000.0;
    
    fill_deterministic(&mut weights);

    // ── Metal Implementation ─────────────────────────────────────────────────
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("Skipping Metal parity test: No Metal device found.");
        return;
    };
    let queue = device.newCommandQueue().unwrap();
    let source = include_str!("../src/kernels.metal");
    let library = device.newLibraryWithSource_options_error(
        &objc2_foundation::NSString::from_str(source),
        None,
    ).expect("Metal kernel compile failed");

    let mut metal_model = LlamaMetalModel::new(weights.meta.clone(), device.clone(), queue, library);
    metal_model.upload_weights(&weights).expect("Metal weight upload failed");

    // Metal needs KV cache buffers
    let kv_bytes_per_token = weights.meta.kv_bytes_per_token(KvCacheType::Fp32);
    let _runtime = MetalMemoryRuntime::new(MetalRuntimeConfig {
        model_name: "test".to_string(),
        memory: MemoryRuntimeConfig::cpu(kv_bytes_per_token, 16, 256, n_layer, 32),
        quantization: weights.meta.quantization,
        tensor_parallel_degree: 1,
    }).unwrap();

    let slot_mapping = device.newBufferWithLength_options(
        (16 * 4) as _, 
        objc2_metal::MTLResourceOptions::StorageModeShared
    ).unwrap();
    let k_pool = device.newBufferWithLength_options(1024*1024, objc2_metal::MTLResourceOptions::StorageModeShared).unwrap();
    let v_pool = device.newBufferWithLength_options(1024*1024, objc2_metal::MTLResourceOptions::StorageModeShared).unwrap();

    let token = 1;
    let pos = 0;
    let metal_logits = metal_model.run(
        token, pos, velo_core::slot_manager::SlotId(0),
        &slot_mapping, &k_pool, &v_pool,
        16, 16, KvCacheType::Fp32,
    ).expect("Metal forward failed");

    // ── CPU Reference ────────────────────────────────────────────────────────
    // We move weights here
    let mut cpu_model = LlamaCpuModel::new(weights);
    let cpu_logits = cpu_model.forward_one(token, pos).expect("CPU forward failed");

    // ── Comparison ───────────────────────────────────────────────────────────
    assert_eq!(cpu_logits.len(), metal_logits.len());
    
    let mut max_diff = 0.0f32;
    for (i, (&c, &m)) in cpu_logits.iter().zip(metal_logits.iter()).enumerate() {
        let diff = (c - m).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(diff < 5e-3, "Logit mismatch at index {}: CPU={}, Metal={}, diff={}", i, c, m, diff);
    }
    
    println!("Llama-3 8B Parity Verified! Max logit diff: {}", max_diff);
}

fn fill_deterministic(weights: &mut WeightStore) {
    let n_floats = weights.data.len() / 4;
    for i in 0..n_floats {
        let val = if i % 2 == 0 {
            0.001 * (1.0 + (i % 10) as f32)
        } else {
            -0.0005 * (1.0 + (i % 7) as f32)
        };
        let bytes = val.to_le_bytes();
        weights.data[i*4 .. (i+1)*4].copy_from_slice(&bytes);
    }
}
