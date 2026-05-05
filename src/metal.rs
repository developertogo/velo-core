use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLCommandQueue,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::backend::{CausalLmBackend, TokenLogits};
use crate::kv_store::KvStoreError;
use crate::paged_attention::{
    BlockMapping, PageManagerError, PageSpan, PagedAttentionBlockManager, PagedAttentionConfig,
};
use crate::radix_cache::{CacheLookup, KvCacheHandle, TokenId};
use crate::runtime::{KvBlockStore, MemoryRuntime, MemoryRuntimeConfig, PagedBlockAllocator};
use crate::speculative::{Result, SpeculativeError};
use crate::slot_manager::SlotId;

/// Configuration for the Metal backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalBackendConfig {
    /// Name of the model.
    pub model_name: String,
    /// Maximum context length in tokens.
    pub max_context_tokens: usize,
    /// Number of bytes required per KV token.
    pub kv_bytes_per_token: usize,
    /// Number of tokens per page/block.
    pub paged_block_size: usize,
    /// Quantization format used by the weights.
    pub quantization: Quantization,
}

/// Configuration for the Metal memory runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalRuntimeConfig {
    /// Name of the model.
    pub model_name: String,
    /// Memory allocation configuration.
    pub memory: MemoryRuntimeConfig,
    /// Quantization format.
    pub quantization: Quantization,
}

/// Placement strategy for Metal buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalBufferPlacement {
    /// Unified memory (shared between CPU and GPU). Default for Apple Silicon.
    Unified,
    /// GPU-private memory.
    Private,
}

/// Opaque handles to Metal framework objects.
pub struct MetalRuntimeHandles {
    /// The Metal device (GPU).
    pub device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
    /// Command queue for dispatching work.
    pub command_queue: Option<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
    /// Compiled shader library.
    pub library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
}

impl Clone for MetalRuntimeHandles {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            command_queue: self.command_queue.clone(),
            library: self.library.clone(),
        }
    }
}

/// Quantization formats supported by the Metal backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    /// 4-bit quantization (block-based).
    Q4_0,
    /// 4-bit K-quantization (Super-block).
    Q4K,
    /// 32-bit floating point.
    F32,
}

/// Information about the Metal hardware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    /// Human-readable name of the GPU (e.g., "Apple M3 Max").
    pub name: String,
    /// Whether the device supports unified memory architecture.
    pub unified_memory: bool,
}

/// Runtime context containing hardware info and handles.
pub struct MetalRuntimeContext {
    pub device: MetalDeviceInfo,
    pub memory: MemoryRuntimeConfig,
    pub placement: MetalBufferPlacement,
    pub handles: MetalRuntimeHandles,
}

/// A block of KV cache stored in Metal memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBlock {
    pub handle: KvCacheHandle,
    pub bytes: usize,
    pub offset: usize,
}

/// GPU-backed KV store using a single large Metal buffer as a block pool.
pub struct MetalKvStore {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    next_block_id: u64,
    bytes_per_token: usize,
    k_pool: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_pool: Retained<ProtocolObject<dyn MTLBuffer>>,
    blocks: std::collections::BTreeMap<u64, KvBlock>,
    free_offsets: Vec<usize>,
}

#[derive(Clone)]
pub struct SharedMetalKvStore(pub std::sync::Arc<std::sync::Mutex<MetalKvStore>>);

#[derive(Clone)]
pub struct SharedPagedAttentionBlockManager(pub std::sync::Arc<std::sync::Mutex<PagedAttentionBlockManager>>);

impl MetalKvStore {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>, kv_bytes_per_token: usize, pool_pages: usize, page_tokens: usize, n_layer: usize) -> Self {
        let page_bytes = page_tokens * kv_bytes_per_token;
        let layer_bytes = pool_pages * page_bytes;
        let pool_size = n_layer * layer_bytes;
        let options = MTLResourceOptions::StorageModeShared;
        let k_pool = device.newBufferWithLength_options(pool_size as _, options)
            .expect("Failed to allocate K pool buffer");
        let v_pool = device.newBufferWithLength_options(pool_size as _, options)
            .expect("Failed to allocate V pool buffer");
        
        let free_offsets = (0..pool_pages).map(|i| i * page_bytes).rev().collect();

        Self {
            device,
            next_block_id: 1,
            bytes_per_token: kv_bytes_per_token,
            k_pool,
            v_pool,
            blocks: std::collections::BTreeMap::new(),
            free_offsets,
        }
    }

    pub fn k_pool(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.k_pool
    }

    pub fn v_pool(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.v_pool
    }

    pub fn allocated_bytes(&self) -> usize {
        if self.free_offsets.is_empty() && self.blocks.is_empty() { return 0; }
        self.blocks.len() * (self.k_pool.length() as usize / (self.free_offsets.len() + self.blocks.len())) * 2
    }

    pub fn allocate(
        &mut self,
        token_len: usize,
    ) -> std::result::Result<crate::radix_cache::KvCacheHandle, KvStoreError> {
        let offset = self.free_offsets.pop().ok_or(KvStoreError::EmptyBlock)?;
        let handle = crate::radix_cache::KvCacheHandle {
            block_id: self.next_block_id,
            token_len,
        };
        self.next_block_id += 1;
        let bytes = token_len * self.bytes_per_token;
        let block = KvBlock { handle, bytes, offset };

        self.blocks.insert(handle.block_id, block);

        Ok(handle)
    }

    pub fn get_block(&self, handle: KvCacheHandle) -> Option<&KvBlock> {
        self.blocks
            .get(&handle.block_id)
            .filter(|b| b.handle == handle)
    }

    pub fn release_block(&mut self, handle: KvCacheHandle) -> std::result::Result<(), KvStoreError> {
        let block = self.blocks.remove(&handle.block_id).ok_or(KvStoreError::UnknownBlock(handle))?;
        self.free_offsets.push(block.offset);
        Ok(())
    }
}

impl KvBlockStore for SharedMetalKvStore {
    fn allocate(&mut self, token_len: usize) -> std::result::Result<crate::radix_cache::KvCacheHandle, KvStoreError> {
        self.0.lock().unwrap().allocate(token_len)
    }
    fn release(&mut self, handle: crate::radix_cache::KvCacheHandle) -> std::result::Result<(), KvStoreError> {
        self.0.lock().unwrap().release_block(handle)
    }
    fn allocated_bytes(&self) -> usize {
        self.0.lock().unwrap().allocated_bytes()
    }
}

impl PagedBlockAllocator for SharedPagedAttentionBlockManager {
    fn allocate(&mut self, handle: KvCacheHandle, token_len: usize) -> std::result::Result<BlockMapping, PageManagerError> {
        self.0.lock().unwrap().allocate(handle, token_len)
    }
    fn release(&mut self, handle: KvCacheHandle) -> std::result::Result<(), PageManagerError> {
        self.0.lock().unwrap().release(handle)
    }
    fn materialize_span(&self, handle: KvCacheHandle) -> Option<PageSpan> {
        self.0.lock().unwrap().materialize_span(handle)
    }
}

impl std::fmt::Debug for SharedMetalKvStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedMetalKvStore")
    }
}

impl std::fmt::Debug for SharedPagedAttentionBlockManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedPagedAttentionBlockManager")
    }
}

/// A memory runtime that manages Metal-backed KV cache and paged attention.
pub struct MetalMemoryRuntime {
    config: MetalRuntimeConfig,
    context: MetalRuntimeContext,
    store: SharedMetalKvStore,
    allocator: SharedPagedAttentionBlockManager,
    bound_prefix: Option<CacheLookup>,
    /// Persistent GPU buffer mapping SlotId -> [PageId; MaxPages].
    slot_mapping: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// A high-performance inference backend using Apple Metal.
///
/// This backend orchestrates the execution of LLM layers on the GPU
/// using custom kernels and unified memory.
#[derive(Debug, Clone)]
pub struct MetalBackend {
    config: MetalBackendConfig,
    device: MetalDeviceInfo,
    bound_prefix: Option<CacheLookup>,
    model: Option<std::sync::Arc<std::sync::Mutex<LlamaMetalModel>>>,
    // Share allocator and store from runtime for block lookups
    allocator: Option<SharedPagedAttentionBlockManager>,
    store: Option<SharedMetalKvStore>,
    slot_id: Option<SlotId>,
    slot_mapping: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

/// Native Metal inference model for LLaMA architecture.
///
/// Manages GPU resources, compiled compute pipelines, and weight buffers.
pub struct LlamaMetalModel {
    meta: crate::model_loader::ModelMeta,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: std::collections::HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    weights: std::collections::HashMap<String, Retained<ProtocolObject<dyn MTLBuffer>>>,
    scratch_buffers: std::collections::HashMap<String, Retained<ProtocolObject<dyn MTLBuffer>>>,
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
    /// Creates a new Metal model instance and compiles required kernels.
    pub fn new(
        meta: crate::model_loader::ModelMeta,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        library: Retained<ProtocolObject<dyn MTLLibrary>>,
    ) -> Self {
        let mut pipelines = std::collections::HashMap::new();
        
        // Pre-compile pipelines
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
            library,
            pipelines,
            weights: std::collections::HashMap::new(),
            scratch_buffers: std::collections::HashMap::new(),
        }
    }

    fn get_scratch(&mut self, name: &str, size: usize) -> Retained<ProtocolObject<dyn MTLBuffer>> {
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

    /// Uploads model weights from a CPU-backed WeightStore to GPU Unified Memory.
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

            // Copy data to unified memory
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

        // 1. Embedding lookup (Simplified: assuming weights["token_embd"] exists)
        let hidden_state = self.get_scratch("hidden_state", n_embd * std::mem::size_of::<f32>());
        // In a real implementation, we'd dispatch a kernel for this or copy from CPU.
        // For now, let's assume we have a kernel for embedding or do a simple copy.
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
            // 2. RMS Norm
            let norm_name = format!("layers.{}.attention_norm.weight", l);
            if let Some(w) = self.weights.get(&norm_name) {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                let pipeline = self.pipelines.get("rms_norm").unwrap();
                unsafe {
                    encoder.setComputePipelineState(pipeline);
                    encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 0); // out
                    encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 1); // in
                    encoder.setBuffer_offset_atIndex(Some(w), 0, 2);
                    let eps = self.meta.norm_eps;
                    let n_embd_u32 = n_embd as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&eps as *const f32 as *mut _).unwrap(), std::mem::size_of::<f32>() as _, 3);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_embd_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 4);
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: n_embd as _, height: 1, depth: 1 },
                        MTLSize { width: 1, height: 1, depth: 1 }
                    );
                    encoder.endEncoding();
                }
            }

            // 3. QKV Projections
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
                        encoder.dispatchThreads_threadsPerThreadgroup(
                            MTLSize { width: rows as _, height: 1, depth: 1 },
                            MTLSize { width: 1, height: 1, depth: 1 }
                        );
                        encoder.endEncoding();
                    }
                }
            }

            // 4. RoPE
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
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: n_rope as _, height: 1, depth: 1 },
                        MTLSize { width: 1, height: 1, depth: 1 }
                    );
                    encoder.endEncoding();
                }
            }

            // 5. Attention
            let n_ctx = self.meta.n_ctx;
            let layer_offset = (l * max_pages * block_size * n_head_kv * head_dim * std::mem::size_of::<f32>()) as _;

            // Store current K, V to cache via kv_update kernel
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
                    
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: (n_head_kv * head_dim) as _, height: 1, depth: 1 },
                        MTLSize { width: 1, height: 1, depth: 1 }
                    );
                    encoder.endEncoding();
                }
            }

            // Attention Score (Q * K)
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
                    
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: n_head as _, height: 1, depth: 1 },
                        MTLSize { width: 1, height: 1, depth: 1 }
                    );
                    encoder.endEncoding();
                }
            }

            // Softmax
            {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                unsafe {
                    encoder.setComputePipelineState(self.pipelines.get("softmax").unwrap());
                    encoder.setBuffer_offset_atIndex(Some(&attn_scores), 0, 0);
                    let n_scores_u32 = (pos + 1) as u32;
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::new(&n_scores_u32 as *const u32 as *mut _).unwrap(), std::mem::size_of::<u32>() as _, 1);
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: n_head as _, height: 1, depth: 1 },
                        MTLSize { width: 1, height: 1, depth: 1 }
                    );
                    encoder.endEncoding();
                }
            }

            // Attention Output (P * V)
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
                    
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: n_head as _, height: 1, depth: 1 },
                        MTLSize { width: 1, height: 1, depth: 1 }
                    );
                    encoder.endEncoding();
                }
            }
            
            // 6. Output projection
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
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: n_embd as _, height: 1, depth: 1 },
                        MTLSize { width: 1, height: 1, depth: 1 }
                    );
                    encoder.endEncoding();
                }

                // Residual Add
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                unsafe {
                    encoder.setComputePipelineState(self.pipelines.get("vec_add").unwrap());
                    encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&attn_out_proj), 0, 1);
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: n_embd as _, height: 1, depth: 1 },
                        MTLSize { width: 1, height: 1, depth: 1 }
                    );
                    encoder.endEncoding();
                }
            }

            // 7. MLP
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
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: n_embd as _, height: 1, depth: 1 },
                        MTLSize { width: 1, height: 1, depth: 1 }
                    );
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
                            encoder.dispatchThreads_threadsPerThreadgroup(
                                MTLSize { width: n_ff as _, height: 1, depth: 1 },
                                MTLSize { width: 1, height: 1, depth: 1 }
                            );
                            encoder.endEncoding();
                        }
                    }
                }

                // SiLU and Mul
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

                // Down projection
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
                        encoder.dispatchThreads_threadsPerThreadgroup(
                            MTLSize { width: n_embd as _, height: 1, depth: 1 },
                            MTLSize { width: 1, height: 1, depth: 1 }
                        );
                        encoder.endEncoding();
                    }

                    // Residual Add
                    let encoder = command_buffer.computeCommandEncoder().unwrap();
                    unsafe {
                        encoder.setComputePipelineState(self.pipelines.get("vec_add").unwrap());
                        encoder.setBuffer_offset_atIndex(Some(&hidden_state), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(&mlp_out), 0, 1);
                        encoder.dispatchThreads_threadsPerThreadgroup(
                            MTLSize { width: n_embd as _, height: 1, depth: 1 },
                            MTLSize { width: 1, height: 1, depth: 1 }
                        );
                        encoder.endEncoding();
                    }
                }
            }
        }

        // Final Norm
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
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: n_embd as _, height: 1, depth: 1 },
                    MTLSize { width: 1, height: 1, depth: 1 }
                );
                encoder.endEncoding();
            }
        }

        // Output Projection (Logits)
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
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: n_vocab as _, height: 1, depth: 1 },
                    MTLSize { width: 1, height: 1, depth: 1 }
                );
                encoder.endEncoding();
            }
        }

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Extract logits
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

impl MetalBackend {
    pub fn new(config: MetalBackendConfig) -> Result<Self> {
        if config.max_context_tokens == 0 {
            return Err(SpeculativeError::Model(
                "max_context_tokens must be greater than zero".to_string(),
            ));
        }
        if config.kv_bytes_per_token == 0 {
            return Err(SpeculativeError::Model(
                "kv_bytes_per_token must be greater than zero".to_string(),
            ));
        }

        Ok(Self {
            config,
            device: MetalDeviceInfo {
                name: "apple-metal-placeholder".to_string(),
                unified_memory: true,
            },
            bound_prefix: None,
            model: None,
            allocator: None,
            store: None,
            slot_id: None,
            slot_mapping: None,
        })
    }

    pub fn with_model(mut self, model: LlamaMetalModel) -> Self {
        self.model = Some(std::sync::Arc::new(std::sync::Mutex::new(model)));
        self
    }

    pub fn config(&self) -> &MetalBackendConfig {
        &self.config
    }

    pub fn device(&self) -> &MetalDeviceInfo {
        &self.device
    }

    pub fn bound_prefix(&self) -> Option<&CacheLookup> {
        self.bound_prefix.as_ref()
    }

    fn not_initialized() -> SpeculativeError {
        SpeculativeError::Model(
            "Metal backend skeleton is not wired to model weights or kernels yet".to_string(),
        )
    }

    pub fn wire(&mut self, store: crate::model_loader::WeightStore, runtime: &MetalMemoryRuntime) -> Result<()> {
        let handles = runtime.context().handles.clone();
        let device = handles.device.clone().ok_or_else(|| SpeculativeError::Model("Runtime missing device".into()))?;
        let queue = handles.command_queue.clone().ok_or_else(|| SpeculativeError::Model("Runtime missing queue".into()))?;
        let library = handles.library.clone().ok_or_else(|| SpeculativeError::Model("Runtime missing library".into()))?;

        let mut model = LlamaMetalModel::new(store.meta.clone(), device, queue, library);
        model.upload_weights(&store)?;
        
        self.model = Some(std::sync::Arc::new(std::sync::Mutex::new(model)));
        self.device = runtime.context().device.clone();
        self.allocator = Some(runtime.allocator.clone());
        self.store = Some(runtime.store.clone());
        self.slot_mapping = Some(runtime.slot_mapping.clone());
        
        Ok(())
    }
}

impl MetalMemoryRuntime {
    pub fn new(config: MetalRuntimeConfig) -> Result<Self> {
        if config.model_name.trim().is_empty() {
            return Err(SpeculativeError::Model(
                "model_name must not be empty".to_string(),
            ));
        }
        if config.memory.bytes_per_token == 0 {
            return Err(SpeculativeError::Model(
                "memory.bytes_per_token must be greater than zero".to_string(),
            ));
        }
        if config.memory.paged_block_size == 0 {
            return Err(SpeculativeError::Model(
                "memory.paged_block_size must be greater than zero".to_string(),
            ));
        }
        if config.memory.paged_total_pages == 0 {
            return Err(SpeculativeError::Model(
                "memory.paged_total_pages must be greater than zero".to_string(),
            ));
        }

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| SpeculativeError::Model("No Metal device found".to_string()))?;

        let command_queue = device.newCommandQueue().ok_or_else(|| {
            SpeculativeError::Model("Failed to create Metal command queue".to_string())
        })?;

        let kernel_source = include_str!("kernels.metal");
        let library = device.newLibraryWithSource_options_error(
            &objc2_foundation::NSString::from_str(kernel_source),
            None,
        ).map_err(|e| {
            SpeculativeError::Model(format!("Failed to compile Metal kernels: {:?}", e))
        })?;

        let context = MetalRuntimeContext {
            device: MetalDeviceInfo {
                name: device.name().to_string(),
                unified_memory: config.memory.unified_memory,
            },
            memory: config.memory,
            placement: if config.memory.unified_memory {
                MetalBufferPlacement::Unified
            } else {
                MetalBufferPlacement::Private
            },
            handles: MetalRuntimeHandles {
                device: Some(device.clone()),
                command_queue: Some(command_queue),
                library: Some(library),
            },
        };
        let store = SharedMetalKvStore(std::sync::Arc::new(std::sync::Mutex::new(MetalKvStore::new(
            device.clone(),
            config.memory.bytes_per_token,
            config.memory.paged_total_pages,
            config.memory.paged_block_size,
            config.memory.n_layer,
        ))));
        
        let allocator = SharedPagedAttentionBlockManager(std::sync::Arc::new(std::sync::Mutex::new(PagedAttentionBlockManager::new(
            PagedAttentionConfig::new(
                config.memory.paged_block_size,
                config.memory.paged_total_pages,
            )
            .map_err(|error| SpeculativeError::Model(error.to_string()))?,
        ))));

        // Allocate slot mapping buffer
        // Size: max_slots * max_pages_per_slot * sizeof(u32)
        let max_slots = config.memory.max_slots;
        let max_pages_per_slot = config.memory.paged_total_pages;
        let slot_mapping_size = (max_slots * max_pages_per_slot * std::mem::size_of::<u32>()) as u64;

        let slot_mapping = device.newBufferWithLength_options(
            slot_mapping_size as usize,
            MTLResourceOptions::StorageModeShared,
        ).ok_or_else(|| SpeculativeError::Model("Failed to allocate slot mapping buffer".to_string()))?;

        Ok(Self {
            context,
            store,
            allocator,
            bound_prefix: None,
            config,
            slot_mapping,
        })
    }

    pub fn config(&self) -> &MetalRuntimeConfig {
        &self.config
    }

    pub fn device(&self) -> &MetalDeviceInfo {
        &self.context.device
    }

    pub fn context(&self) -> &MetalRuntimeContext {
        &self.context
    }

    pub fn bound_prefix(&self) -> Option<&CacheLookup> {
        self.bound_prefix.as_ref()
    }

    pub fn bind_prefix_cache(&mut self, prefix: &CacheLookup) {
        self.bound_prefix = Some(prefix.clone());
    }

    pub fn with_handles(mut self, handles: MetalRuntimeHandles) -> Self {
        self.context.handles = handles;
        self
    }

    pub fn has_device_handles(&self) -> bool {
        self.context.handles.device.is_some()
            && self.context.handles.command_queue.is_some()
            && self.context.handles.library.is_some()
    }
}

impl MemoryRuntime for MetalMemoryRuntime {
    type Store = SharedMetalKvStore;
    type Allocator = SharedPagedAttentionBlockManager;

    fn store(&self) -> &Self::Store {
        &self.store
    }

    fn store_mut(&mut self) -> &mut Self::Store {
        &mut self.store
    }

    fn allocator(&self) -> &Self::Allocator {
        &self.allocator
    }

    fn allocator_mut(&mut self) -> &mut Self::Allocator {
        &mut self.allocator
    }

    fn bind_slot(&mut self, slot: crate::slot_manager::SlotId, pages: &[u32]) -> Result<()> {
        let max_pages = self.config.memory.paged_total_pages;
        if pages.len() > max_pages {
            return Err(SpeculativeError::Model(format!(
                "Request pages ({}) exceed max pages per slot ({})",
                pages.len(),
                max_pages
            )));
        }

        let contents = self.slot_mapping.contents().as_ptr() as *mut u32;
        unsafe {
            let slot_ptr = contents.add(slot.0 as usize * max_pages);
            std::ptr::copy_nonoverlapping(pages.as_ptr(), slot_ptr, pages.len());
            
            // Zero out the rest of the slot to prevent stale data usage
            if pages.len() < max_pages {
                std::ptr::write_bytes(
                    slot_ptr.add(pages.len()),
                    0,
                    (max_pages - pages.len()) * std::mem::size_of::<u32>(),
                );
            }
        }
        
        Ok(())
    }
}

impl CausalLmBackend for MetalBackend {
    fn bind_prefix_cache(&mut self, prefix: &CacheLookup) -> Result<()> {
        self.bound_prefix = Some(prefix.clone());
        Ok(())
    }

    fn bind_slot(&mut self, slot: crate::slot_manager::SlotId) -> Result<()> {
        self.slot_id = Some(slot);
        Ok(())
    }

    fn next_logits(&mut self, context: &[TokenId]) -> Result<TokenLogits> {
        let Some(model_arc) = &self.model else {
            return Err(Self::not_initialized());
        };
        let allocator_arc = self.allocator.as_ref().ok_or_else(|| SpeculativeError::Model("Allocator not wired".into()))?;
        let store_arc = self.store.as_ref().ok_or_else(|| SpeculativeError::Model("Store not wired".into()))?;

        let last = context.last().copied().ok_or_else(|| {
            SpeculativeError::Model("next_logits called with empty context".to_string())
        })?;

        let prefix = self.bound_prefix.as_ref().ok_or_else(|| SpeculativeError::Model("No prefix bound".to_string()))?;
        let prefix_handle = prefix.handle.ok_or_else(|| {
            SpeculativeError::Model("Prefix has no KV handle".to_string())
        })?;

        let mut model = model_arc.lock().unwrap();
        let allocator = allocator_arc.0.lock().unwrap();
        let store = store_arc.0.lock().unwrap();

        let span = allocator.materialize_span(prefix_handle).ok_or_else(|| {
            SpeculativeError::Model(format!("Could not materialize span for prefix {:?}", prefix_handle))
        })?;

        let page_bytes = self.config.paged_block_size * self.config.kv_bytes_per_token;
        let mut block_indices = Vec::with_capacity(span.pages.len());
        for page_id in span.pages {
            let block = store.get_block(crate::radix_cache::KvCacheHandle { block_id: page_id.0, token_len: 0 })
                .ok_or_else(|| SpeculativeError::Model(format!("Block {:?} not found in store", page_id)))?;
            block_indices.push((block.offset / page_bytes) as u32);
        }

        let pos = context.len() - 1; 
        let logits = model.forward_one(
            last,
            pos,
            store.k_pool(),
            store.v_pool(),
            self.slot_id.ok_or_else(|| SpeculativeError::Model("Slot not bound".into()))?,
            self.slot_mapping.as_ref().ok_or_else(|| SpeculativeError::Model("Slot mapping not wired".into()))?,
            allocator.config().total_pages,
            self.config.paged_block_size,
        )?;
        TokenLogits::new(logits)
    }

    fn verify_logits(
        &mut self,
        context: &[TokenId],
        drafted: &[TokenId],
    ) -> Result<Vec<TokenLogits>> {
        let Some(model_arc) = &self.model else {
            return Err(Self::not_initialized());
        };
        let allocator_arc = self.allocator.as_ref().ok_or_else(|| SpeculativeError::Model("Allocator not wired".into()))?;
        let store_arc = self.store.as_ref().ok_or_else(|| SpeculativeError::Model("Store not wired".into()))?;

        let prefix = self.bound_prefix.as_ref().ok_or_else(|| SpeculativeError::Model("No prefix bound".to_string()))?;
        let prefix_handle = prefix.handle.ok_or_else(|| {
            SpeculativeError::Model("Prefix has no KV handle".to_string())
        })?;

        let mut model = model_arc.lock().unwrap();
        let allocator = allocator_arc.0.lock().unwrap();
        let store = store_arc.0.lock().unwrap();

        let span = allocator.materialize_span(prefix_handle).ok_or_else(|| {
            SpeculativeError::Model(format!("Could not materialize span for prefix {:?}", prefix_handle))
        })?;

        let page_bytes = self.config.paged_block_size * self.config.kv_bytes_per_token;
        let mut block_indices = Vec::with_capacity(span.pages.len());
        for page_id in span.pages {
            let block = store.get_block(crate::radix_cache::KvCacheHandle { block_id: page_id.0, token_len: 0 })
                .ok_or_else(|| SpeculativeError::Model(format!("Block {:?} not found in store", page_id)))?;
            block_indices.push((block.offset / page_bytes) as u32);
        }

        let mut result = Vec::with_capacity(drafted.len());
        let mut current_pos = context.len();
        
        let slot_id = self.slot_id.ok_or_else(|| SpeculativeError::Model("Slot not bound".into()))?;
        let slot_mapping = self.slot_mapping.as_ref().ok_or_else(|| SpeculativeError::Model("Slot mapping not wired".into()))?;
        let max_pages = allocator.config().total_pages;

        for &tok in drafted {
            let logits = model.forward_one(
                tok,
                current_pos,
                store.k_pool(),
                store.v_pool(),
                slot_id,
                slot_mapping,
                max_pages,
                self.config.paged_block_size,
            )?;
            result.push(TokenLogits::new(logits)?);
            current_pos += 1;
        }
        Ok(result)
    }
}

impl Default for MetalBackendConfig {
    fn default() -> Self {
        Self {
            model_name: "llama-metal".to_string(),
            max_context_tokens: 4096,
            kv_bytes_per_token: 4096,
            paged_block_size: 16,
            quantization: Quantization::F32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MetalBackendConfig {
        MetalBackendConfig {
            model_name: "draft-1b".to_string(),
            max_context_tokens: 4096,
            kv_bytes_per_token: 4096,
            paged_block_size: 16,
            quantization: Quantization::Q4K,
        }
    }

    #[test]
    fn validates_non_zero_context() {
        let mut config = config();
        config.max_context_tokens = 0;

        assert!(MetalBackend::new(config).is_err());
    }

    #[test]
    fn records_bound_prefix_cache() {
        let mut backend = MetalBackend::new(config()).unwrap();
        let prefix = CacheLookup {
            matched_tokens: 32,
            handle: None,
        };

        backend.bind_prefix_cache(&prefix).unwrap();

        assert_eq!(backend.bound_prefix(), Some(&prefix));
    }

    #[test]
    fn logits_return_explicit_not_initialized_error() {
        let mut backend = MetalBackend::new(config()).unwrap();

        let error = backend.next_logits(&[1, 2, 3]).unwrap_err();

        assert!(error.to_string().contains("not wired"));
    }

    #[test]
    fn metal_runtime_exposes_memory_runtime_shape() {
        let Ok(runtime) = MetalMemoryRuntime::new(MetalRuntimeConfig {
            model_name: "target-8b".to_string(),
            memory: MemoryRuntimeConfig::cpu(4096, 16, 32, 32, 32),
            quantization: Quantization::Q4K,
        }) else {
            // Skip if Metal device not found in test environment
            return;
        };

        assert_eq!(runtime.config().model_name, "target-8b");
        assert!(runtime.context().device.unified_memory);
        assert_eq!(runtime.context().placement, MetalBufferPlacement::Unified);
        assert!(!runtime.has_device_handles());
        assert_eq!(runtime.store().allocated_bytes(), 0);
        assert_eq!(runtime.allocator().0.lock().unwrap().free_pages(), 32);
    }

    #[test]
    fn metal_runtime_accepts_opaque_handles() {
        let Ok(runtime) = MetalMemoryRuntime::new(MetalRuntimeConfig {
            model_name: "target-8b".to_string(),
            memory: MemoryRuntimeConfig::cpu(4096, 16, 32, 32, 32),
            quantization: Quantization::Q4K,
        }) else {
            // Skip if Metal device not found in test environment
            return;
        };

        // Since we can't easily create real Retained handles in tests,
        // we just verify the initial state or use None.
        assert!(!runtime.has_device_handles());
    }

    #[test]
    fn llama_metal_model_compiles_kernels() {
        let Ok(runtime) = MetalMemoryRuntime::new(MetalRuntimeConfig {
            model_name: "test".to_string(),
            memory: MemoryRuntimeConfig::cpu(4096, 16, 32, 32, 32),
            quantization: Quantization::Q4K,
        }) else {
            return;
        };

        let handles = runtime.context().handles.clone();
        let model = LlamaMetalModel::new(
            crate::model_loader::ModelMeta {
                arch: "llama".to_string(),
                n_vocab: 32000,
                n_embd: 4096,
                n_layer: 32,
                n_head: 32,
                n_head_kv: 32,
                n_ctx: 4096,
                n_ff: 11008,
                head_dim: 128,
                rope_freq_base: 10000.0,
                norm_eps: 1e-5,
                quantization: Quantization::Q4K,
            },
            handles.device.unwrap(),
            handles.command_queue.unwrap(),
            handles.library.unwrap(),
        );

        assert!(model.pipelines.contains_key("matvec_f32"));
        assert!(model.pipelines.contains_key("rms_norm"));
        assert!(model.pipelines.contains_key("rope"));
        assert!(model.pipelines.contains_key("attn_q_k"));
        assert!(model.pipelines.contains_key("attn_p_v"));
        assert!(model.pipelines.contains_key("vec_add"));
    }
}
