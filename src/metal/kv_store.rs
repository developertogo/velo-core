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

unsafe impl Send for MetalKvStore {}
unsafe impl Sync for MetalKvStore {}

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

    /// Returns a reference to the K pool Metal buffer.
    pub fn k_pool(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.k_pool
    }

    /// Returns a reference to the V pool Metal buffer.
    pub fn v_pool(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.v_pool
    }

    /// Returns the total number of bytes currently allocated in the pools.
    pub fn allocated_bytes(&self) -> usize {
        self.blocks.values().map(|b| b.bytes).sum()
    }

    /// Allocates a new block from the pool for a given sequence length.
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

    /// Returns a reference to a block given its handle.
    pub fn get_block(&self, handle: KvCacheHandle) -> Option<&KvBlock> {
        self.blocks
            .get(&handle.block_id)
            .filter(|b| b.handle == handle)
    }

    /// Releases a block back to the pool.
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

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    #[test]
    fn test_metal_kv_store_allocation() {
        let device = match MTLCreateSystemDefaultDevice() {
            Some(d) => d,
            None => {
                eprintln!("Skipping Metal test: No device available");
                return;
            }
        };

        let kv_bytes_per_token = 128;
        let pool_pages = 10;
        let page_tokens = 16;
        let n_layer = 1;

        let mut store = MetalKvStore::new(device, kv_bytes_per_token, pool_pages, page_tokens, n_layer);
        
        assert_eq!(store.allocated_bytes(), 0);

        let handle1 = store.allocate(16).unwrap();
        assert_eq!(handle1.block_id, 1);
        assert_eq!(handle1.token_len, 16);

        let block1 = store.get_block(handle1).unwrap();
        assert_eq!(block1.handle, handle1);
        assert_eq!(block1.bytes, 16 * 128);

        let handle2 = store.allocate(16).unwrap();
        assert_eq!(handle2.block_id, 2);

        store.release_block(handle1).unwrap();
        assert!(store.get_block(handle1).is_none());

        // Allocate again, should reuse
        let handle3 = store.allocate(16).unwrap();
        assert_eq!(handle3.block_id, 3);
        
        // Exceed capacity
        for _ in 0..8 {
            store.allocate(16).unwrap();
        }
        assert!(store.allocate(16).is_err());
    }

    #[test]
    fn test_shared_metal_kv_store() {
        let device = match MTLCreateSystemDefaultDevice() {
            Some(d) => d,
            None => return,
        };

        let store = MetalKvStore::new(device, 128, 5, 16, 1);
        let mut shared = SharedMetalKvStore(Arc::new(Mutex::new(store)));

        let handle = shared.allocate(16).unwrap();
        assert!(shared.allocated_bytes() > 0);
        
        shared.release(handle).unwrap();
        assert!(format!("{:?}", shared).contains("SharedMetalKvStore"));
    }
}
