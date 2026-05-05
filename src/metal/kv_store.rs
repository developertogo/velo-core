use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::kv_store::KvStoreError;
use crate::radix_cache::KvCacheHandle;
use crate::runtime::KvBlockStore;

/// A block of KV cache stored in Metal memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBlock {
    pub handle: KvCacheHandle,
    pub bytes: usize,
    pub offset: usize,
}

/// GPU-backed KV store using a single large Metal buffer as a block pool.
pub struct MetalKvStore {
    next_block_id: u64,
    bytes_per_token: usize,
    k_pool: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_pool: Retained<ProtocolObject<dyn MTLBuffer>>,
    blocks: BTreeMap<u64, KvBlock>,
    free_offsets: Vec<usize>,
}

#[derive(Clone)]
pub struct SharedMetalKvStore(pub Arc<Mutex<MetalKvStore>>);

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
            next_block_id: 1,
            bytes_per_token: kv_bytes_per_token,
            k_pool,
            v_pool,
            blocks: BTreeMap::new(),
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
    ) -> Result<KvCacheHandle, KvStoreError> {
        let offset = self.free_offsets.pop().ok_or(KvStoreError::EmptyBlock)?;
        let handle = KvCacheHandle {
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

    pub fn release_block(&mut self, handle: KvCacheHandle) -> Result<(), KvStoreError> {
        let block = self.blocks.remove(&handle.block_id).ok_or(KvStoreError::UnknownBlock(handle))?;
        self.free_offsets.push(block.offset);
        Ok(())
    }
}

impl KvBlockStore for SharedMetalKvStore {
    fn allocate(&mut self, token_len: usize) -> Result<KvCacheHandle, KvStoreError> {
        self.0.lock().unwrap().allocate(token_len)
    }
    fn release(&mut self, handle: KvCacheHandle) -> Result<(), KvStoreError> {
        self.0.lock().unwrap().release_block(handle)
    }
    fn allocated_bytes(&self) -> usize {
        self.0.lock().unwrap().allocated_bytes()
    }
}

impl std::fmt::Debug for SharedMetalKvStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedMetalKvStore")
    }
}
