use std::collections::BTreeMap;

use crate::radix_cache::KvCacheHandle;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBlock {
    pub handle: KvCacheHandle,
    pub bytes: usize,
}

pub trait KvStore {
    fn allocate(&mut self, token_len: usize) -> Result<KvCacheHandle, KvStoreError>;

    fn get(&self, handle: KvCacheHandle) -> Option<&KvBlock>;

    fn release(&mut self, handle: KvCacheHandle) -> Result<(), KvStoreError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvStoreError {
    EmptyBlock,
    UnknownBlock(KvCacheHandle),
}

#[derive(Debug, Clone)]
pub struct InMemoryKvStore {
    next_block_id: u64,
    bytes_per_token: usize,
    blocks: BTreeMap<u64, KvBlock>,
}

impl InMemoryKvStore {
    pub fn new(bytes_per_token: usize) -> Self {
        Self {
            next_block_id: 1,
            bytes_per_token,
            blocks: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.blocks.values().map(|block| block.bytes).sum()
    }
}

impl KvStore for InMemoryKvStore {
    fn allocate(&mut self, token_len: usize) -> Result<KvCacheHandle, KvStoreError> {
        if token_len == 0 {
            return Err(KvStoreError::EmptyBlock);
        }

        let handle = KvCacheHandle {
            block_id: self.next_block_id,
            token_len,
        };
        self.next_block_id += 1;

        let block = KvBlock {
            handle,
            bytes: token_len * self.bytes_per_token,
        };
        self.blocks.insert(handle.block_id, block);

        Ok(handle)
    }

    fn get(&self, handle: KvCacheHandle) -> Option<&KvBlock> {
        self.blocks
            .get(&handle.block_id)
            .filter(|block| block.handle == handle)
    }

    fn release(&mut self, handle: KvCacheHandle) -> Result<(), KvStoreError> {
        let Some(block) = self.blocks.remove(&handle.block_id) else {
            return Err(KvStoreError::UnknownBlock(handle));
        };

        if block.handle != handle {
            self.blocks.insert(block.handle.block_id, block);
            return Err(KvStoreError::UnknownBlock(handle));
        }

        Ok(())
    }
}

impl std::fmt::Display for KvStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBlock => write!(formatter, "KV block must contain at least one token"),
            Self::UnknownBlock(handle) => write!(
                formatter,
                "unknown KV block {} with {} tokens",
                handle.block_id, handle.token_len
            ),
        }
    }
}

impl std::error::Error for KvStoreError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_blocks_with_stable_handles() {
        let mut store = InMemoryKvStore::new(128);

        let first = store.allocate(4).unwrap();
        let second = store.allocate(2).unwrap();

        assert_eq!(first.block_id, 1);
        assert_eq!(second.block_id, 2);
        assert_eq!(store.get(first).unwrap().bytes, 512);
        assert_eq!(store.allocated_bytes(), 768);
    }

    #[test]
    fn releases_known_blocks() {
        let mut store = InMemoryKvStore::new(64);
        let handle = store.allocate(3).unwrap();

        store.release(handle).unwrap();

        assert!(store.get(handle).is_none());
        assert!(store.is_empty());
    }

    #[test]
    fn rejects_empty_allocations() {
        let mut store = InMemoryKvStore::new(64);

        assert_eq!(store.allocate(0), Err(KvStoreError::EmptyBlock));
    }
}
