use std::collections::{BTreeMap, VecDeque};

use crate::radix_cache::KvCacheHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedAttentionConfig {
    pub block_size: usize,
    pub page_tokens: usize,
    pub total_pages: usize,
    pub unified_memory: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSpan {
    pub pages: Vec<PageId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockMapping {
    pub handle: KvCacheHandle,
    pub pages: Vec<PageId>,
    pub token_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageManagerError {
    InvalidBlockSize,
    OutOfPages { requested: usize, available: usize },
    UnknownHandle(KvCacheHandle),
}

#[derive(Debug)]
pub struct PagedAttentionBlockManager {
    config: PagedAttentionConfig,
    free_pages: VecDeque<PageId>,
    mappings: BTreeMap<u64, BlockMapping>,
}

impl PagedAttentionConfig {
    pub fn new(block_size: usize, total_pages: usize) -> Result<Self, PageManagerError> {
        if block_size == 0 {
            return Err(PageManagerError::InvalidBlockSize);
        }

        Ok(Self {
            block_size,
            page_tokens: block_size,
            total_pages,
            unified_memory: true,
        })
    }
}

impl PagedAttentionBlockManager {
    pub fn new(config: PagedAttentionConfig) -> Self {
        let mut free_pages = VecDeque::new();
        for idx in 0..config.total_pages as u64 {
            free_pages.push_back(PageId(idx));
        }

        Self {
            config,
            free_pages,
            mappings: BTreeMap::new(),
        }
    }

    pub fn config(&self) -> &PagedAttentionConfig {
        &self.config
    }

    pub fn mapped_pages(&self, handle: KvCacheHandle) -> Option<&[PageId]> {
        self.mappings.get(&handle.block_id).map(|mapping| mapping.pages.as_slice())
    }

    pub fn pages_for_tokens(&self, token_len: usize) -> usize {
        token_len.div_ceil(self.config.block_size)
    }

    pub fn allocate(
        &mut self,
        handle: KvCacheHandle,
        token_len: usize,
    ) -> Result<BlockMapping, PageManagerError> {
        let required_pages = self.pages_for_tokens(token_len);
        let available = self.free_pages.len();
        if required_pages > available {
            return Err(PageManagerError::OutOfPages {
                requested: required_pages,
                available,
            });
        }

        let mut pages = Vec::with_capacity(required_pages);
        for _ in 0..required_pages {
            let page = self.free_pages.pop_front().expect("capacity was checked");
            pages.push(page);
        }

        let mapping = BlockMapping {
            handle,
            pages,
            token_len,
        };
        self.mappings.insert(handle.block_id, mapping.clone());
        Ok(mapping)
    }

    pub fn release(&mut self, handle: KvCacheHandle) -> Result<(), PageManagerError> {
        let Some(mapping) = self.mappings.remove(&handle.block_id) else {
            return Err(PageManagerError::UnknownHandle(handle));
        };

        for page in mapping.pages {
            self.free_pages.push_back(page);
        }
        Ok(())
    }

    pub fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    pub fn materialize_span(&self, handle: KvCacheHandle) -> Option<PageSpan> {
        let mapping = self.mappings.get(&handle.block_id)?;
        Some(PageSpan {
            pages: mapping.pages.clone(),
        })
    }
}

impl std::fmt::Display for PageManagerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBlockSize => write!(formatter, "block size must be greater than zero"),
            Self::OutOfPages { requested, available } => write!(
                formatter,
                "requested {requested} pages but only {available} are available"
            ),
            Self::UnknownHandle(handle) => write!(
                formatter,
                "unknown KV handle {} with {} tokens",
                handle.block_id, handle.token_len
            ),
        }
    }
}

impl std::error::Error for PageManagerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_pages_from_a_fixed_budget() {
        let config = PagedAttentionConfig::new(16, 4).unwrap();
        let mut manager = PagedAttentionBlockManager::new(config);
        let handle = KvCacheHandle {
            block_id: 1,
            token_len: 24,
        };

        let mapping = manager.allocate(handle, 24).unwrap();

        assert_eq!(mapping.pages.len(), 2);
        assert_eq!(manager.free_pages(), 2);
        assert!(manager.mapped_pages(handle).is_some());
        assert_eq!(manager.materialize_span(handle).unwrap().pages.len(), 2);
    }

    #[test]
    fn release_returns_pages_to_pool() {
        let config = PagedAttentionConfig::new(8, 2).unwrap();
        let mut manager = PagedAttentionBlockManager::new(config);
        let handle = KvCacheHandle {
            block_id: 2,
            token_len: 8,
        };

        manager.allocate(handle, 8).unwrap();
        manager.release(handle).unwrap();

        assert_eq!(manager.free_pages(), 2);
    }

    #[test]
    fn rejects_oversized_allocations() {
        let config = PagedAttentionConfig::new(32, 1).unwrap();
        let mut manager = PagedAttentionBlockManager::new(config);
        let handle = KvCacheHandle {
            block_id: 3,
            token_len: 64,
        };

        let error = manager.allocate(handle, 64).unwrap_err();

        assert!(matches!(
            error,
            PageManagerError::OutOfPages {
                requested: 2,
                available: 1
            }
        ));
    }
}
