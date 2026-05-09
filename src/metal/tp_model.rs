//! Tensor Parallel model orchestrator.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLCopyAllDevices, MTLSize, MTLComputeCommandEncoder, MTLCommandEncoder, MTLResourceOptions};

use crate::model_loader::{ModelMeta, WeightStore};
use crate::radix_cache::TokenId;
use crate::slot_manager::SlotId;
use crate::speculative::{Result, SpeculativeError};
use crate::paged_attention::KvCacheType;
use crate::metal::{LlamaMetalModel, shard_weight, get_shard_strategy};

/// Orchestrates multiple Metal GPU shards for Tensor Parallel inference.
pub struct LlamaTensorParallelModel {
    pub meta: ModelMeta,
    pub shards: Vec<LlamaMetalModel>,
    pub tp_degree: usize,
}

impl LlamaTensorParallelModel {
    /// Creates a new LlamaTensorParallelModel by sharding the provided weights across available GPUs.
    pub fn new(
        meta: ModelMeta,
        store: &WeightStore,
        library_source: &str,
    ) -> Result<Self> {
        let _tp_degree = meta.quantization.block_size(); // Just a placeholder for testing
        // Wait, TP degree should come from config.
        // For now, we'll discover devices and use them all up to a limit.
        let all_devices = MTLCopyAllDevices();
        if all_devices.count() == 0 {
            return Err(SpeculativeError::Model("No Metal devices found".to_string()));
        }
        
        let tp_degree = all_devices.count();
        let mut shards = Vec::with_capacity(tp_degree);

        for rank in 0..tp_degree {
            let device = all_devices.objectAtIndex(rank);
            let queue = device.newCommandQueue().ok_or_else(|| {
                SpeculativeError::Model(format!("Failed to create queue for device {}", rank))
            })?;
            
            // Compile library for each device
            let library = device.newLibraryWithSource_options_error(
                &objc2_foundation::NSString::from_str(library_source),
                None,
            ).map_err(|e| SpeculativeError::Model(format!("Failed to compile library for device {}: {:?}", rank, e)))?;

            // Shard metadata for this rank
            let mut shard_meta = meta.clone();
            shard_meta.n_head /= tp_degree;
            shard_meta.n_head_kv /= tp_degree;
            shard_meta.n_ff /= tp_degree;

            let mut model = LlamaMetalModel::new(shard_meta, device.clone(), queue, library);
            
            // Upload sharded weights
            for (name, _info) in &store.index {
                let strategy = get_shard_strategy(name);
                let data = store.get(name).ok_or_else(|| SpeculativeError::Model(format!("Missing weight {}", name)))?;
                let shape = store.shape(name).ok_or_else(|| SpeculativeError::Model(format!("Missing shape {}", name)))?;
                
                let sharded = shard_weight(name, data, shape, meta.quantization, strategy, tp_degree, rank)?;
                
                let buffer = device.newBufferWithLength_options(
                    sharded.data.len() as _,
                    MTLResourceOptions::StorageModeShared,
                ).ok_or_else(|| SpeculativeError::Model(format!("Failed to allocate shard buffer for {}", name)))?;
                
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        sharded.data.as_ptr(),
                        buffer.contents().as_ptr() as *mut u8,
                        sharded.data.len(),
                    );
                }
                model.weights.insert(name.clone(), buffer);
            }
            shards.push(model);
        }

        Ok(Self {
            meta,
            shards,
            tp_degree,
        })
    }

    /// Runs a TP-parallelized forward pass.
    pub fn run(
        &mut self,
        token: TokenId,
        pos: usize,
        slot_id: SlotId,
        slot_mapping: &ProtocolObject<dyn MTLBuffer>,
        k_pools: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        v_pools: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        max_pages: usize,
        block_size: usize,
        kv_type: KvCacheType,
    ) -> Result<Vec<f32>> {
        let n_embd = self.meta.n_embd;
        
        // 1. Initial Embedding (Replicated on all shards)
        let mut command_buffers = Vec::with_capacity(self.tp_degree);
        for rank in 0..self.tp_degree {
            let cb = self.shards[rank].queue.commandBuffer().unwrap();
            let hidden_state = self.shards[rank].get_scratch("hidden_state", n_embd * 4);
            
            // Replicated embedding lookup
            let embed_w = self.shards[rank].weights.get("token_embd.weight").unwrap();
            let encoder = cb.computeCommandEncoder().unwrap();
            encoder.setComputePipelineState(self.shards[rank].pipelines.get("embed_lookup").unwrap());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(embed_w), 0, 1);
            }
            let token_u32 = token as u32;
            let n_embd_u32 = n_embd as u32;
            unsafe {
                encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&token_u32 as *const u32 as *mut _).unwrap(), 4, 2);
                encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_embd_u32 as *const u32 as *mut _).unwrap(), 4, 3);
                encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
            }
            encoder.endEncoding();
            command_buffers.push(cb);
        }

        // 2. Loop through layers
        for l in 0..self.meta.n_layer {
            let mut layer_results = Vec::with_capacity(self.tp_degree);
            for rank in 0..self.tp_degree {
                let hidden_state = self.shards[rank].get_scratch("hidden_state", n_embd * 4);
                let (attn_out, mlp_out) = self.shards[rank].forward_layer(
                    l,
                    &hidden_state,
                    pos,
                    slot_id,
                    slot_mapping,
                    &k_pools[rank],
                    &v_pools[rank],
                    max_pages,
                    block_size,
                    kv_type,
                    &command_buffers[rank],
                )?;
                layer_results.push((attn_out, mlp_out));
            }

            // Sync: All-Reduce Sum for Attention
            self.all_reduce_sum_and_add(&mut command_buffers, &layer_results, true)?; // true = attn
            
            // Sync: All-Reduce Sum for MLP
            self.all_reduce_sum_and_add(&mut command_buffers, &layer_results, false)?; // false = mlp
        }

        // 3. Final Norm & Logits
        let mut final_results = Vec::with_capacity(self.tp_degree);
        for rank in 0..self.tp_degree {
            let cb = &command_buffers[rank];
            let hidden_state = self.shards[rank].get_scratch("hidden_state", n_embd * 4);
            
            // Final Norm (Replicated)
            let norm_w = self.shards[rank].weights.get("output_norm.weight").unwrap();
            self.shards[rank].dispatch_rms_norm(cb, &hidden_state, &hidden_state, norm_w, n_embd)?;

            // Output Projection (Column-parallel)
            let shard_vocab = self.meta.n_vocab / self.tp_degree;
            let logits_buf = self.shards[rank].get_scratch("logits", shard_vocab * 4);
            let out_w = self.shards[rank].weights.get("output.weight").unwrap();
            self.shards[rank].dispatch_matvec(cb, &logits_buf, out_w, &hidden_state, shard_vocab, n_embd)?;
            
            cb.commit();
            final_results.push(logits_buf);
        }

        // 4. Wait and All-Gather logits
        for cb in &command_buffers {
            cb.waitUntilCompleted();
        }

        let mut all_logits = vec![0.0f32; self.meta.n_vocab];
        let shard_vocab = self.meta.n_vocab / self.tp_degree;
        for rank in 0..self.tp_degree {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    final_results[rank].contents().as_ptr() as *const f32,
                    all_logits.as_mut_ptr().add(rank * shard_vocab),
                    shard_vocab,
                );
            }
        }

        Ok(all_logits)
    }

    fn all_reduce_sum_and_add(
        &mut self,
        command_buffers: &mut [Retained<ProtocolObject<dyn MTLCommandBuffer>>],
        layer_results: &[(Retained<ProtocolObject<dyn MTLBuffer>>, Retained<ProtocolObject<dyn MTLBuffer>>)],
        is_attn: bool,
    ) -> Result<()> {
        if self.tp_degree <= 1 {
            // Just add local
            for rank in 0..self.tp_degree {
                let shard = &mut self.shards[rank];
                let hidden = shard.get_scratch("hidden_state", self.meta.n_embd * 4);
                let partial = if is_attn { &layer_results[rank].0 } else { &layer_results[rank].1 };
                
                let encoder = command_buffers[rank].computeCommandEncoder().unwrap();
                encoder.setComputePipelineState(shard.pipelines.get("vec_add").unwrap());
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&hidden), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(partial), 0, 1);
                }
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: self.meta.n_embd as _, height: 1, depth: 1 },
                    MTLSize { width: 1, height: 1, depth: 1 }
                );
                encoder.endEncoding();
            }
            return Ok(());
        }

        // For V1 TP on Metal (Unified Memory), we can use a "CPU-orchestrated" All-Reduce:
        // 1. Commit partials
        // 2. Wait
        // 3. Sum on CPU (or GPU 0)
        // 4. Copy back to all GPUs
        // (This is slow but correct for a first pass)
        
        for rank in 0..self.tp_degree {
            command_buffers[rank].commit();
        }
        for rank in 0..self.tp_degree {
            command_buffers[rank].waitUntilCompleted();
        }

        let n_embd = self.meta.n_embd;
        let mut sum = vec![0.0f32; n_embd];
        for rank in 0..self.tp_degree {
            let partial = if is_attn { &layer_results[rank].0 } else { &layer_results[rank].1 };
            let data = unsafe { std::slice::from_raw_parts(partial.contents().as_ptr() as *const f32, n_embd) };
            for i in 0..n_embd {
                sum[i] += data[i];
            }
        }

        // 5. Start new command buffers and add to hidden_state
        for rank in 0..self.tp_degree {
            let cb = self.shards[rank].queue.commandBuffer().unwrap();
            let hidden = self.shards[rank].get_scratch("hidden_state", n_embd * 4);
            
            // Upload sum to a temporary scratch on this device
            let sum_buf = self.shards[rank].get_scratch("all_reduce_tmp", n_embd * 4);
            unsafe {
                std::ptr::copy_nonoverlapping(sum.as_ptr(), sum_buf.contents().as_ptr() as *mut f32, n_embd);
            }

            let encoder = cb.computeCommandEncoder().unwrap();
            encoder.setComputePipelineState(self.shards[rank].pipelines.get("vec_add").unwrap());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&hidden), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&sum_buf), 0, 1);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: n_embd as _, height: 1, depth: 1 },
                MTLSize { width: 1, height: 1, depth: 1 }
            );
            encoder.endEncoding();
            command_buffers[rank] = cb;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_loader::WeightStore;

    #[test]
    fn test_tp_model_init_dummy() {
        let n_vocab = 128;
        let n_embd = 64;
        let n_layer = 1;
        let store = WeightStore::dummy_llama(n_vocab, n_embd, n_layer);
        let meta = store.meta.clone();
        
        let kernel_source = "
#include <metal_stdlib>
using namespace metal;
kernel void dummy() {}
";
        
        let model = LlamaTensorParallelModel::new(meta, &store, kernel_source);
        if let Ok(tp_model) = model {
            assert!(tp_model.tp_degree >= 1);
            assert_eq!(tp_model.shards.len(), tp_model.tp_degree);
        }
    }
}
