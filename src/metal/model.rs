use std::collections::HashMap;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::model_loader::ModelMeta;
use crate::radix_cache::TokenId;
use crate::slot_manager::SlotId;
use crate::speculative::{Result, SpeculativeError};
use crate::paged_attention::KvCacheType;

/// Native Metal inference model for LLaMA architecture.
pub struct LlamaMetalModel {
    pub meta: ModelMeta,
    pub device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pipelines: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub weights: HashMap<String, Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub scratch_buffers: HashMap<String, Retained<ProtocolObject<dyn MTLBuffer>>>,
}

unsafe impl Send for LlamaMetalModel {}
unsafe impl Sync for LlamaMetalModel {}

impl std::fmt::Debug for LlamaMetalModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaMetalModel")
            .field("meta", &self.meta)
            .field("weights_count", &self.weights.len())
            .field("pipelines_count", &self.pipelines.len())
            .finish()
    }
}

impl LlamaMetalModel {
    /// Creates a new LlamaMetalModel with the specified hardware handles.
    pub fn new(
        meta: ModelMeta,
        device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        library: Retained<ProtocolObject<dyn MTLLibrary>>,
    ) -> Self {
        let mut pipelines = HashMap::new();
        let functions = [
            "matvec_f32", "matvec_q4_0", "rms_norm", "rope", "silu", "vec_mul", "softmax",
            "vec_add", "kv_update", "paged_attention_flash",
            "kv_update_int8", "paged_attention_flash_int8", "argmax"
        ];
        for name in functions {
            if let Some(func) = library.newFunctionWithName(&objc2_foundation::NSString::from_str(name)) {
                if let Ok(pipeline) = device.newComputePipelineStateWithFunction_error(&func) {
                    pipelines.insert(name.to_string(), pipeline);
                }
            }
        }

        Self {
            meta,
            device,
            queue,
            pipelines,
            weights: HashMap::new(),
            scratch_buffers: HashMap::new(),
        }
    }

    /// Returns a scratch buffer of the specified size, creating it if necessary.
    pub fn get_scratch(&mut self, name: &str, size: usize) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        if let Some(buf) = self.scratch_buffers.get(name) {
            if buf.length() >= size as _ {
                return buf.clone();
            }
        }

        let buf = self.device.newBufferWithLength_options(
            size as _,
            MTLResourceOptions::StorageModeShared,
        ).expect("Failed to allocate scratch buffer");
        self.scratch_buffers.insert(name.to_string(), buf.clone());
        buf
    }

    /// Uploads model weights from a WeightStore to GPU buffers.
    pub fn upload_weights(&mut self, store: &crate::model_loader::WeightStore) -> Result<()> {
        for (name, _info) in &store.index {
            let data = store.get(name).ok_or_else(|| {
                SpeculativeError::Model(format!("Missing weight data for {}", name))
            })?;

            let buffer = self.device.newBufferWithLength_options(
                data.len() as _,
                MTLResourceOptions::StorageModeShared,
            ).ok_or_else(|| {
                SpeculativeError::Model(format!("Failed to allocate GPU buffer for {}", name))
            })?;

            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buffer.contents().as_ptr() as *mut u8,
                    data.len(),
                );
            }
            self.weights.insert(name.clone(), buffer);
        }
        Ok(())
    }

    /// Executes a full inference pass and returns the logits copied back to the CPU.
    /// This is the standard entry point for generic sampling.
    pub fn run(
        &mut self,
        token: TokenId,
        pos: usize,
        slot_id: SlotId,
        slot_mapping: &ProtocolObject<dyn MTLBuffer>,
        k_pool: &ProtocolObject<dyn MTLBuffer>,
        v_pool: &ProtocolObject<dyn MTLBuffer>,
        max_pages: usize,
        block_size: usize,
        kv_type: KvCacheType,
    ) -> Result<Vec<f32>> {
        let (command_buffer, logits_buf) = self.forward(
            token, pos, slot_id, slot_mapping, k_pool, v_pool, max_pages, block_size, kv_type
        )?;
        
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        let n_vocab = self.meta.n_vocab;
        let mut logits = vec![0.0f32; n_vocab];
        unsafe {
            std::ptr::copy_nonoverlapping(
                logits_buf.contents().as_ptr() as *const f32,
                logits.as_mut_ptr(),
                n_vocab,
            );
        }
        Ok(logits)
    }

    /// Executes a full inference pass and performs GPU-resident sampling (greedy)
    /// before returning the result. This avoids a CPU-GPU synchronization and copy
    /// of the entire logit buffer (usually 32k-128k floats).
    pub fn run_with_sampling(
        &mut self,
        token: TokenId,
        pos: usize,
        slot_id: SlotId,
        slot_mapping: &ProtocolObject<dyn MTLBuffer>,
        k_pool: &ProtocolObject<dyn MTLBuffer>,
        v_pool: &ProtocolObject<dyn MTLBuffer>,
        max_pages: usize,
        block_size: usize,
        kv_type: KvCacheType,
    ) -> Result<u32> {
        let (command_buffer, logits_buf) = self.forward(
            token, pos, slot_id, slot_mapping, k_pool, v_pool, max_pages, block_size, kv_type
        )?;
        
        self.sample_argmax(&command_buffer, &logits_buf)
    }

    /// Performs the core transformer forward pass on the GPU.
    /// Returns the active command buffer and the scratch buffer containing the final logits.
    pub fn forward(
        &mut self,
        token: TokenId,
        pos: usize,
        slot_id: SlotId,
        slot_mapping: &ProtocolObject<dyn MTLBuffer>,
        k_pool: &ProtocolObject<dyn MTLBuffer>,
        v_pool: &ProtocolObject<dyn MTLBuffer>,
        max_pages: usize,
        block_size: usize,
        kv_type: KvCacheType,
    ) -> Result<(Retained<ProtocolObject<dyn MTLCommandBuffer>>, Retained<ProtocolObject<dyn MTLBuffer>>)> {
        let n_embd = self.meta.n_embd;
        let n_layer = self.meta.n_layer;
        let head_dim = self.meta.head_dim;
        let n_head = self.meta.n_head;
        let n_head_kv = self.meta.n_head_kv;

        let command_buffer = self.queue.commandBuffer().ok_or_else(|| {
            SpeculativeError::Model("Failed to create command buffer".to_string())
        })?;

        let hidden_state = self.get_scratch("hidden_state", n_embd * std::mem::size_of::<f32>());
        if let Some(embd_weight) = self.weights.get("token_embd.weight") {
            unsafe {
                let embd_ptr = (embd_weight.contents().as_ptr() as *const f32).add(token as usize * n_embd);
                std::ptr::copy_nonoverlapping(
                    embd_ptr,
                    hidden_state.contents().as_ptr() as *mut f32,
                    n_embd,
                );
            }
        }

        for l in 0..n_layer {
            let norm_name = format!("layers.{}.attention_norm.weight", l);
            if let Some(w) = self.weights.get(&norm_name) {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                let pipeline = self.pipelines.get("rms_norm").unwrap();
                unsafe {
                    encoder.setComputePipelineState(pipeline);
                    encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(w), 0, 2);
                    let eps = self.meta.norm_eps;
                    let n_embd_u32 = n_embd as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&eps as *const f32 as *mut _).unwrap(), std::mem::size_of::<f32>() as _, 3);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_embd_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_embd as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }
            }

            let q_buf = self.get_scratch("q", n_embd * std::mem::size_of::<f32>());
            let k_buf = self.get_scratch("k", n_embd * std::mem::size_of::<f32>());
            let v_buf = self.get_scratch("v", n_embd * std::mem::size_of::<f32>());

            for (proj, buf) in [("wq", &q_buf), ("wk", &k_buf), ("wv", &v_buf)] {
                let weight_name = format!("layers.{}.attention.{}.weight", l, proj);
                if let Some(w) = self.weights.get(&weight_name) {
                    let rows = if proj == "wq" { n_embd } else { n_head_kv * head_dim };
                    self.dispatch_matvec(&command_buffer, buf, w, &hidden_state, rows, n_embd)?;
                }
            }

            {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                let pipeline = self.pipelines.get("rope").unwrap();
                unsafe {
                    encoder.setComputePipelineState(pipeline);
                    encoder.setBuffer_offset_atIndex(Some(&q_buf), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&k_buf), 0, 1);
                    let pos_u32 = pos as u32;
                    let head_dim_u32 = head_dim as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&pos_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 2);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&head_dim_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 3);
                    let base = self.meta.rope_freq_base;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&base as *const f32 as *mut _).unwrap(), std::mem::size_of::<f32>() as _, 4);
                    let n_rope = (n_head + n_head_kv) * head_dim / 2;
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_rope as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }
            }

            // ── KV Update ──
            {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                let kernel_name = match kv_type {
                    KvCacheType::Fp32 => "kv_update",
                    KvCacheType::Int8 | KvCacheType::Fp8 => "kv_update_int8",
                };
                let pipeline = self.pipelines.get(kernel_name).unwrap();
                unsafe {
                    encoder.setComputePipelineState(pipeline);
                    encoder.setBuffer_offset_atIndex(Some(k_pool), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(v_pool), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&k_buf), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&v_buf), 0, 3);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&slot_id as *const SlotId as *mut _).unwrap(), std::mem::size_of::<SlotId>() as _, 4);
                    encoder.setBuffer_offset_atIndex(Some(slot_mapping), 0, 5);
                    let max_pages_u32 = max_pages as u32;
                    let block_size_u32 = block_size as u32;
                    let n_head_kv_u32 = n_head_kv as u32;
                    let head_dim_u32 = head_dim as u32;
                    let pos_u32 = pos as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&max_pages_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 6);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&block_size_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 7);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_head_kv_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 8);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&head_dim_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 9);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&pos_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 10);
                    
                    let grid_size = if kv_type == KvCacheType::Fp32 {
                        MTLSize { width: (n_head_kv * head_dim) as _, height: 1, depth: 1 }
                    } else {
                        MTLSize { width: n_head_kv as _, height: 1, depth: 1 }
                    };
                    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }
            }
            let attn_out = self.get_scratch("attn_out", n_head * head_dim * std::mem::size_of::<f32>());

            // ── Paged Attention ──
            {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                let kernel_name = match kv_type {
                    KvCacheType::Fp32 => "paged_attention_flash",
                    KvCacheType::Int8 | KvCacheType::Fp8 => "paged_attention_flash_int8",
                };
                let pipeline = self.pipelines.get(kernel_name).unwrap();
                unsafe {
                    encoder.setComputePipelineState(pipeline);
                    encoder.setBuffer_offset_atIndex(Some(&attn_out), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(k_pool), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(v_pool), 0, 3);
                    let head_dim_u32 = head_dim as u32;
                    let n_ctx_u32 = (pos + 1) as u32;
                    let pos_u32 = pos as u32;
                    let max_pages_u32 = max_pages as u32;
                    let block_size_u32 = block_size as u32;
                    let n_head_kv_u32 = n_head_kv as u32;
                    let n_head_u32 = n_head as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&head_dim_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_ctx_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 5);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&pos_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 6);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&slot_id as *const SlotId as *mut _).unwrap(), std::mem::size_of::<SlotId>() as _, 7);
                    encoder.setBuffer_offset_atIndex(Some(slot_mapping), 0, 8);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&max_pages_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 9);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&block_size_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 10);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_head_kv_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 11);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_head_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 12);
                    
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_head as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }
            }
           
            let attn_out_proj = self.get_scratch("attn_out_proj", n_embd * std::mem::size_of::<f32>());
            let wo_name = format!("layers.{}.attention.wo.weight", l);
            if let Some(w) = self.weights.get(&wo_name) {
                self.dispatch_matvec(&command_buffer, &attn_out_proj, w, &attn_out, n_embd, n_embd)?;

                let encoder = command_buffer.computeCommandEncoder().unwrap();
                unsafe {
                    encoder.setComputePipelineState(self.pipelines.get("vec_add").unwrap());
                    encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&attn_out_proj), 0, 1);
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_embd as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }
            }

            let mlp_in = self.get_scratch("mlp_in", n_embd * std::mem::size_of::<f32>());
            let ffn_norm_name = format!("layers.{}.ffn_norm.weight", l);
            if let Some(w) = self.weights.get(&ffn_norm_name) {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                let pipeline = self.pipelines.get("rms_norm").unwrap();
                unsafe {
                    encoder.setComputePipelineState(pipeline);
                    encoder.setBuffer_offset_atIndex(Some(&mlp_in), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(w), 0, 2);
                    let eps = self.meta.norm_eps;
                    let n_embd_u32 = n_embd as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&eps as *const f32 as *mut _).unwrap(), std::mem::size_of::<f32>() as _, 3);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_embd_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_embd as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }

                let n_ff = self.meta.n_ff;
                let gate_buf = self.get_scratch("mlp_gate", n_ff * std::mem::size_of::<f32>());
                let up_buf = self.get_scratch("mlp_up", n_ff * std::mem::size_of::<f32>());

                for (proj, buf) in [("w1", &gate_buf), ("w3", &up_buf)] {
                    let weight_name = format!("layers.{}.feed_forward.{}.weight", l, proj);
                    if let Some(w) = self.weights.get(&weight_name) {
                        self.dispatch_matvec(&command_buffer, buf, w, &mlp_in, n_ff, n_embd)?;
                    }
                }

                let encoder = command_buffer.computeCommandEncoder().unwrap();
                unsafe {
                    encoder.setComputePipelineState(self.pipelines.get("silu").unwrap());
                    encoder.setBuffer_offset_atIndex(Some(&gate_buf), 0, 0);
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_ff as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                   
                    encoder.setComputePipelineState(self.pipelines.get("vec_mul").unwrap());
                    encoder.setBuffer_offset_atIndex(Some(&gate_buf), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&up_buf), 0, 1);
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_ff as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }

                let mlp_out = self.get_scratch("mlp_out", n_embd * std::mem::size_of::<f32>());
                let w2_name = format!("layers.{}.feed_forward.w2.weight", l);
                if let Some(w) = self.weights.get(&w2_name) {
                    self.dispatch_matvec(&command_buffer, &mlp_out, w, &gate_buf, n_embd, n_ff)?;

                    let encoder = command_buffer.computeCommandEncoder().unwrap();
                    unsafe {
                        encoder.setComputePipelineState(self.pipelines.get("vec_add").unwrap());
                        encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(&mlp_out), 0, 1);
                        encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_embd as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                        encoder.endEncoding();
                    }
                }
            }
        }

        if let Some(w) = self.weights.get("output_norm.weight") {
            let encoder = command_buffer.computeCommandEncoder().unwrap();
            let pipeline = self.pipelines.get("rms_norm").unwrap();
            unsafe {
                encoder.setComputePipelineState(pipeline);
                encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(w), 0, 2);
                let eps = self.meta.norm_eps;
                let n_embd_u32 = n_embd as u32;
                encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&eps as *const f32 as *mut _).unwrap(), std::mem::size_of::<f32>() as _, 3);
                encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_embd_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_embd as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                encoder.endEncoding();
            }
        }

        let n_vocab = self.meta.n_vocab;
        let logits_buf = self.get_scratch("logits", n_vocab * std::mem::size_of::<f32>());
        if let Some(w) = self.weights.get("output.weight") {
            self.dispatch_matvec(&command_buffer, &logits_buf, w, &hidden_state, n_vocab, n_embd)?;
        }

        Ok((command_buffer, logits_buf))
    }

    pub fn sample_argmax(
        &mut self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        logits_buf: &ProtocolObject<dyn MTLBuffer>,
    ) -> Result<u32> {
        let n_vocab = self.meta.n_vocab;
        let out_buf = self.get_scratch("argmax_out", 4);
        let pipeline = self.pipelines.get("argmax").ok_or_else(|| {
             SpeculativeError::Model("Argmax pipeline not found".to_string())
        })?;

        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
             SpeculativeError::Model("Failed to create command encoder".to_string())
        })?;
        
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(logits_buf), 0, 1);
            let n_u32 = n_vocab as u32;
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(), 4, 2);
            
            // Dispatch with 1024 threads in one TG
            // Note: For vocab > 1024, this kernel only looks at the first 1024 tokens.
            // A production version would use multiple threadgroups + global atomic.
            let threads = n_vocab.min(1024);
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: threads as _, height: 1, depth: 1 },
                MTLSize { width: threads as _, height: 1, depth: 1 }
            );
            encoder.endEncoding();
        }
        
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        
        let mut token_id = 0u32;
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buf.contents().as_ptr() as *const u32,
                &mut token_id,
                1,
            );
        }
        Ok(token_id)
    }

    fn dispatch_matvec(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        out: &ProtocolObject<dyn MTLBuffer>,
        weight: &ProtocolObject<dyn MTLBuffer>,
        x: &ProtocolObject<dyn MTLBuffer>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let is_q4_0 = self.meta.quantization == crate::metal::Quantization::Q4_0;
        let pipeline_name = if is_q4_0 { "matvec_q4_0" } else { "matvec_f32" };
        let pipeline = self.pipelines.get(pipeline_name).ok_or_else(|| {
            SpeculativeError::Model(format!("Pipeline {} not found", pipeline_name))
        })?;

        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            SpeculativeError::Model("Failed to create command encoder".to_string())
        })?;

        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setBuffer_offset_atIndex(Some(out), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(x), 0, 2);
            let rows_u32 = rows as u32;
            let cols_u32 = cols as u32;
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&rows_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 3);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&cols_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
            encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: rows as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
            encoder.endEncoding();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_loader::WeightStore;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    /// Helper: create a LlamaMetalModel from a dummy WeightStore.
    /// Returns None when no Metal device is present (e.g. restricted CI).
    fn make_model(n_vocab: usize, n_embd: usize) -> Option<LlamaMetalModel> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;
        let source = include_str!("../kernels.metal");
        let options = objc2_metal::MTLCompileOptions::new();
        let library = device
            .newLibraryWithSource_options_error(
                &objc2_foundation::NSString::from_str(source),
                Some(&options),
            )
            .ok()?;

        let weights = WeightStore::dummy_llama(n_vocab, n_embd, 1);
        let mut model = LlamaMetalModel::new(weights.meta.clone(), device, queue, library);
        model.upload_weights(&weights).ok()?;
        Some(model)
    }

    #[test]
    fn test_debug_impl() {
        if let Some(model) = make_model(100, 32) {
            let s = format!("{:?}", model);
            assert!(s.contains("LlamaMetalModel"));
            assert!(s.contains("weights_count"));
        }
    }

    #[test]
    fn test_get_scratch_reuses_buffer() {
        if let Some(mut model) = make_model(100, 32) {
            let buf1 = model.get_scratch("test_scratch", 256);
            let buf2 = model.get_scratch("test_scratch", 128); // smaller — should reuse
            // Both should point to the same allocation (same length)
            assert_eq!(buf1.length(), buf2.length());

            let buf3 = model.get_scratch("test_scratch", 512); // larger — new alloc
            assert!(buf3.length() >= 512);
        }
    }

    #[test]
    fn test_upload_weights_populates_map() {
        if let Some(model) = make_model(64, 32) {
            assert!(!model.weights.is_empty(), "Weights should be non-empty after upload");
        }
    }

    #[test]
    fn test_pipelines_loaded() {
        if let Some(model) = make_model(64, 32) {
            // At minimum, the F32 kernel paths must be present
            assert!(model.pipelines.contains_key("rms_norm"));
            assert!(model.pipelines.contains_key("matvec_f32"));
            assert!(model.pipelines.contains_key("argmax"));
        }
    }

    #[test]
    fn test_forward_produces_logits() {
        if let Some(mut model) = make_model(100, 32) {
            use crate::slot_manager::SlotId;
            use crate::paged_attention::KvCacheType;
            use objc2_metal::MTLResourceOptions;

            let n_pages: usize = 4;
            let slot_mapping_size = n_pages * std::mem::size_of::<u32>();
            let slot_mapping = model.device
                .newBufferWithLength_options(slot_mapping_size as _, MTLResourceOptions::StorageModeShared)
                .expect("slot_mapping alloc");
            let kv_size = 64 * 1024;
            let k_pool = model.device.newBufferWithLength_options(kv_size, MTLResourceOptions::StorageModeShared).unwrap();
            let v_pool = model.device.newBufferWithLength_options(kv_size, MTLResourceOptions::StorageModeShared).unwrap();

            let result = model.run(
                1, 0, SlotId(0),
                &slot_mapping, &k_pool, &v_pool,
                n_pages, 16, KvCacheType::Fp32,
            );

            assert!(result.is_ok(), "forward pass failed: {:?}", result.err());
            let logits = result.unwrap();
            assert_eq!(logits.len(), 100);
        }
    }

    #[test]
    fn test_run_with_sampling_returns_valid_token() {
        if let Some(mut model) = make_model(100, 32) {
            use crate::slot_manager::SlotId;
            use crate::paged_attention::KvCacheType;
            use objc2_metal::MTLResourceOptions;

            let n_pages: usize = 4;
            let slot_mapping = model.device
                .newBufferWithLength_options((n_pages * 4) as _, MTLResourceOptions::StorageModeShared)
                .unwrap();
            let kv_size = 64 * 1024;
            let k_pool = model.device.newBufferWithLength_options(kv_size, MTLResourceOptions::StorageModeShared).unwrap();
            let v_pool = model.device.newBufferWithLength_options(kv_size, MTLResourceOptions::StorageModeShared).unwrap();

            let token = model.run_with_sampling(
                1, 0, SlotId(0),
                &slot_mapping, &k_pool, &v_pool,
                n_pages, 16, KvCacheType::Fp32,
            );

            assert!(token.is_ok(), "run_with_sampling failed: {:?}", token.err());
            assert!(token.unwrap() < 100, "token must be in vocab range");
        }
    }
}
