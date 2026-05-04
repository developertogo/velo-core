use crate::kv_store::{InMemoryKvStore, KvStore, KvStoreError};
use crate::paged_attention::{
    BlockMapping, PageManagerError, PageSpan, PagedAttentionBlockManager, PagedAttentionConfig,
};
use crate::radix_cache::KvCacheHandle;

pub trait KvBlockStore {
    fn allocate(&mut self, token_len: usize) -> Result<KvCacheHandle, KvStoreError>;
    fn release(&mut self, handle: KvCacheHandle) -> Result<(), KvStoreError>;
    fn allocated_bytes(&self) -> usize;
}

pub trait PagedBlockAllocator {
    fn allocate(
        &mut self,
        handle: KvCacheHandle,
        token_len: usize,
    ) -> Result<BlockMapping, PageManagerError>;
    fn release(&mut self, handle: KvCacheHandle) -> Result<(), PageManagerError>;
    fn materialize_span(&self, handle: KvCacheHandle) -> Option<PageSpan>;
}

pub trait MemoryRuntime {
    type Store: KvBlockStore;
    type Allocator: PagedBlockAllocator;

    fn store(&self) -> &Self::Store;
    fn store_mut(&mut self) -> &mut Self::Store;
    fn allocator(&self) -> &Self::Allocator;
    fn allocator_mut(&mut self) -> &mut Self::Allocator;
    /// Binds a specific request slot to a sequence of physical page indices.
    ///
    /// This allows the backend to update its persistent page table for the given
    /// slot, enabling O(1) lookups during kernel execution.
    fn bind_slot(&mut self, slot: crate::slot_manager::SlotId, pages: &[u32]) -> Result<(), crate::speculative::SpeculativeError>;
}

/// Configuration for the memory runtime and KV-cache layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRuntimeConfig {
    /// Number of bytes required per KV token (total across all heads).
    pub bytes_per_token: usize,
    /// Number of tokens per page/block.
    pub paged_block_size: usize,
    /// Total number of pages available in the pool.
    pub paged_total_pages: usize,
    /// Number of layers in the model.
    pub n_layer: usize,
    /// Whether to use unified memory (Apple Silicon specific).
    pub unified_memory: bool,
    /// Maximum number of concurrent request slots to support.
    pub max_slots: usize,
}

impl MemoryRuntimeConfig {
    /// Creates a new configuration.
    pub fn new(bytes_per_token: usize, paged_block_size: usize, paged_total_pages: usize, n_layer: usize, max_slots: usize) -> Self {
        Self {
            bytes_per_token,
            paged_block_size,
            paged_total_pages,
            n_layer,
            unified_memory: true,
            max_slots,
        }
    }

    /// Creates a configuration optimized for CPU execution.
    pub fn cpu(bytes_per_token: usize, paged_block_size: usize, paged_total_pages: usize, n_layer: usize, max_slots: usize) -> Self {
        let mut config = Self::new(bytes_per_token, paged_block_size, paged_total_pages, n_layer, max_slots);
        config.unified_memory = false;
        config
    }
}

#[derive(Debug)]
pub struct CpuMemoryRuntime {
    store: InMemoryKvStore,
    allocator: PagedAttentionBlockManager,
}

impl CpuMemoryRuntime {
    pub fn new(config: MemoryRuntimeConfig) -> Result<Self, PageManagerError> {
        Ok(Self {
            store: InMemoryKvStore::new(config.bytes_per_token),
            allocator: PagedAttentionBlockManager::new(PagedAttentionConfig::new(
                config.paged_block_size,
                config.paged_total_pages,
            )?),
        })
    }
}

impl MemoryRuntime for CpuMemoryRuntime {
    type Store = InMemoryKvStore;
    type Allocator = PagedAttentionBlockManager;

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

    fn bind_slot(&mut self, _slot: crate::slot_manager::SlotId, _pages: &[u32]) -> Result<(), crate::speculative::SpeculativeError> {
        Ok(())
    }
}

impl KvBlockStore for InMemoryKvStore {
    fn allocate(&mut self, token_len: usize) -> Result<KvCacheHandle, KvStoreError> {
        KvStore::allocate(self, token_len)
    }

    fn release(&mut self, handle: KvCacheHandle) -> Result<(), KvStoreError> {
        KvStore::release(self, handle)
    }

    fn allocated_bytes(&self) -> usize {
        InMemoryKvStore::allocated_bytes(self)
    }
}

impl PagedBlockAllocator for PagedAttentionBlockManager {
    fn allocate(
        &mut self,
        handle: KvCacheHandle,
        token_len: usize,
    ) -> Result<BlockMapping, PageManagerError> {
        PagedAttentionBlockManager::allocate(self, handle, token_len)
    }

    fn release(&mut self, handle: KvCacheHandle) -> Result<(), PageManagerError> {
        PagedAttentionBlockManager::release(self, handle)
    }

    fn materialize_span(&self, handle: KvCacheHandle) -> Option<PageSpan> {
        PagedAttentionBlockManager::materialize_span(self, handle)
    }
}
