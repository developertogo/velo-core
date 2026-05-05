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

/// Native Metal inference model for LLaMA architecture.
pub struct LlamaMetalModel {
    pub meta: ModelMeta,
    pub device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pipelines: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub weights: HashMap<String, Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub scratch_buffers: HashMap<String, Retained<ProtocolObject<dyn MTLBuffer>>>,
}

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
    pub fn new(
        meta: ModelMeta,
        device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        library: Retained<ProtocolObject<dyn MTLLibrary>>,
    ) -> Self {
        let mut pipelines = HashMap::new();
        let functions = [
            "matvec_f32", "rms_norm", "rope", "silu", "vec_mul", "softmax",
            "attn_q_k", "attn_p_v", "vec_add", "kv_update"
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

    pub fn forward_one(
        &mut self,
        token: TokenId,
        pos: usize,
        k_pool: &ProtocolObject<dyn MTLBuffer>,
        v_pool: &ProtocolObject<dyn MTLBuffer>,
        slot_id: SlotId,
        slot_mapping: &ProtocolObject<dyn MTLBuffer>,
        max_pages: usize,
        block_size: usize,
    ) -> Result<Vec<f32>> {
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
                    let encoder = command_buffer.computeCommandEncoder().unwrap();
                    let pipeline = self.pipelines.get("matvec_f32").unwrap();
                    unsafe {
                        encoder.setComputePipelineState(pipeline);
                        encoder.setBuffer_offset_atIndex(Some(buf), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(w), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 2);
                        let rows = if proj == "wq" { n_embd } else { n_head_kv * head_dim };
                        let rows_u32 = rows as u32;
                        let n_embd_u32 = n_embd as u32;
                        encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&rows_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 3);
                        encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_embd_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                        encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: rows as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                        encoder.endEncoding();
                    }
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

            let n_ctx = self.meta.n_ctx;
            let layer_offset = (l * max_pages * block_size * n_head_kv * head_dim * std::mem::size_of::<f32>()) as _;

            {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                unsafe {
                    encoder.setComputePipelineState(self.pipelines.get("kv_update").unwrap());
                    encoder.setBuffer_offset_atIndex(Some(k_pool), layer_offset, 0);
                    encoder.setBuffer_offset_atIndex(Some(v_pool), layer_offset, 1);
                    encoder.setBuffer_offset_atIndex(Some(&k_buf), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&v_buf), 0, 3);
                    let slot_id_u32 = slot_id.0 as u32;
                    let max_pages_u32 = max_pages as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&slot_id_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                    encoder.setBuffer_offset_atIndex(Some(slot_mapping), 0, 5);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&max_pages_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 6);
                    let block_size_u32 = block_size as u32;
                    let n_head_kv_u32 = n_head_kv as u32;
                    let head_dim_u32 = head_dim as u32;
                    let pos_u32 = pos as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&block_size_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 7);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_head_kv_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 8);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&head_dim_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 9);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&pos_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 10);
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: (n_head_kv * head_dim) as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }
            }

            let attn_scores = self.get_scratch("attn_scores", n_head * n_ctx * std::mem::size_of::<f32>());
            {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                unsafe {
                    encoder.setComputePipelineState(self.pipelines.get("attn_q_k").unwrap());
                    encoder.setBuffer_offset_atIndex(Some(&attn_scores), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(k_pool), layer_offset, 2);
                    let head_dim_u32 = head_dim as u32;
                    let n_ctx_u32 = n_ctx as u32;
                    let pos_u32 = pos as u32;
                    let block_size_u32 = block_size as u32;
                    let n_head_kv_u32 = n_head_kv as u32;
                    let slot_id_u32 = slot_id.0 as u32;
                    let max_pages_u32 = max_pages as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&head_dim_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 3);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_ctx_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&pos_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 5);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&slot_id_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 6);
                    encoder.setBuffer_offset_atIndex(Some(slot_mapping), 0, 7);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&max_pages_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 8);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&block_size_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 9);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_head_kv_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 10);
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_head as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }
            }

            {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                unsafe {
                    encoder.setComputePipelineState(self.pipelines.get("softmax").unwrap());
                    encoder.setBuffer_offset_atIndex(Some(&attn_scores), 0, 0);
                    let n_scores_u32 = (pos + 1) as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_scores_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 1);
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_head as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }
            }

            let attn_out = self.get_scratch("attn_out", n_head * head_dim * std::mem::size_of::<f32>());
            {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                unsafe {
                    encoder.setComputePipelineState(self.pipelines.get("attn_p_v").unwrap());
                    encoder.setBuffer_offset_atIndex(Some(&attn_out), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&attn_scores), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(v_pool), layer_offset, 2);
                    let head_dim_u32 = head_dim as u32;
                    let n_ctx_u32 = n_ctx as u32;
                    let pos_u32 = pos as u32;
                    let block_size_u32 = block_size as u32;
                    let n_head_kv_u32 = n_head_kv as u32;
                    let slot_id_u32 = slot_id.0 as u32;
                    let max_pages_u32 = max_pages as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&head_dim_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 3);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_ctx_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&pos_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 5);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&slot_id_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 6);
                    encoder.setBuffer_offset_atIndex(Some(slot_mapping), 0, 7);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&max_pages_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 8);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&block_size_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 9);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_head_kv_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 10);
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_head as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }
            }
            
            let attn_out_proj = self.get_scratch("attn_out_proj", n_embd * std::mem::size_of::<f32>());
            let wo_name = format!("layers.{}.attention.wo.weight", l);
            if let Some(w) = self.weights.get(&wo_name) {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                let pipeline = self.pipelines.get("matvec_f32").unwrap();
                unsafe {
                    encoder.setComputePipelineState(pipeline);
                    encoder.setBuffer_offset_atIndex(Some(&attn_out_proj), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(w), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&attn_out), 0, 2);
                    let n_embd_u32 = n_embd as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_embd_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 3);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_embd_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                    encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_embd as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                }

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
                        let encoder = command_buffer.computeCommandEncoder().unwrap();
                        let pipeline = self.pipelines.get("matvec_f32").unwrap();
                        unsafe {
                            encoder.setComputePipelineState(pipeline);
                            encoder.setBuffer_offset_atIndex(Some(buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(w), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(&mlp_in), 0, 2);
                            let n_ff_u32 = n_ff as u32;
                            let n_embd_u32 = n_embd as u32;
                            encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_ff_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 3);
                            encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_embd_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                            encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_ff as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                            encoder.endEncoding();
                        }
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
                    let encoder = command_buffer.computeCommandEncoder().unwrap();
                    let pipeline = self.pipelines.get("matvec_f32").unwrap();
                    unsafe {
                        encoder.setComputePipelineState(pipeline);
                        encoder.setBuffer_offset_atIndex(Some(&mlp_out), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(w), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(&gate_buf), 0, 2);
                        let n_embd_u32 = n_embd as u32;
                        let n_ff_u32 = n_ff as u32;
                        encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_embd_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 3);
                        encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_ff_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                        encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_embd as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                        encoder.endEncoding();
                    }

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
            let encoder = command_buffer.computeCommandEncoder().unwrap();
            let pipeline = self.pipelines.get("matvec_f32").unwrap();
            unsafe {
                encoder.setComputePipelineState(pipeline);
                encoder.setBuffer_offset_atIndex(Some(&logits_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(w), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 2);
                let n_vocab_u32 = n_vocab as u32;
                let n_embd_u32 = n_embd as u32;
                encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_vocab_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 3);
                encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_embd_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: n_vocab as _, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
                encoder.endEncoding();
            }
        }

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

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
}
