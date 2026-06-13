//! Paged Attention: Virtual Memory for LLMs
//!
//! Modern LLMs generate text token by token. For each token, the model needs to store
//! some data in the "KV-cache" (Key-Value cache). 
//!
//! **The Problem:** We don't know how long a conversation will be. If we reserve a huge
//! block of memory for every user, we'll run out of memory (VRAM) very quickly.
//!
//! **The Solution:** Paged Attention. Just like your computer's OS uses "pages" to 
//! manage RAM, Velo splits the KV-cache into small, fixed-size chunks called **Pages**.
//!
//! This allows us to:
//! 1. **Reduce Waste**: We only allocate memory as we need it.
//! 2. **Share Memory**: Multiple people can share the same "prefix" (like a long 
//!    instruction prompt) without duplicating it in memory.

use std::collections::{BTreeMap, VecDeque};
use crate::radix_cache::KvCacheHandle;

/// A unique identifier for a physical chunk of memory on the GPU.
/// Think of this as a "house number" in the city of VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u64);

/// The numerical format of the data stored in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default)]
pub enum KvCacheType {
    #[default]
    /// High precision (32-bit floats).
    Fp32,
    /// Compressed (8-bit integers) - saves 4x memory!
    Int8,
    /// Modern floating point (8-bit) - balanced speed and accuracy.
    Fp8,
}

/// Configuration for the memory manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedAttentionConfig {
    /// How many tokens fit in a single memory block.
    pub block_size: usize,
    /// Same as block_size (compatibility alias).
    pub page_tokens: usize,
    /// Total number of pages available in the GPU pool.
    pub total_pages: usize,
    /// Whether the CPU and GPU share the same memory (always true on Apple Silicon).
    pub unified_memory: bool,
    /// The data format (FP32, INT8, etc).
    pub kv_type: KvCacheType,
}

/// A sequence of pages that together store a piece of text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSpan {
    /// The ordered list of pages.
    pub pages: Vec<PageId>,
}

/// A map that connects a high-level "Handle" to physical "Pages".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockMapping {
    /// The high-level ID for this piece of text.
    pub handle: KvCacheHandle,
    /// The physical memory pages where the data is actually stored.
    pub pages: Vec<PageId>,
    /// How many tokens are actually in this block.
    pub token_len: usize,
}

/// Errors that happen when we run out of memory or have invalid settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageManagerError {
    /// You tried to create a block with 0 size.
    InvalidBlockSize,
    /// The GPU is full! No more pages available.
    OutOfPages { requested: usize, available: usize },
    /// You tried to access memory that doesn't exist.
    UnknownHandle(KvCacheHandle),
}

/// PagedAttentionBlockManager: The Parking Garage Manager
///
/// Imagine a parking garage where every car (request) needs a different number of 
/// spots (tokens). Some cars are tiny, some are buses.
///
/// This manager:
/// 1. Keeps a list of "empty spots" (`free_pages`).
/// 2. Hands out spots when a new car arrives (`allocate`).
/// 3. Takes the spots back when the car leaves (`release`).
/// 4. Remembers which car is in which spots (`mappings`).
#[derive(Debug)]
pub struct PagedAttentionBlockManager {
    /// Settings for the manager.
    config: PagedAttentionConfig,
    /// A list of physical pages that aren't being used yet.
    free_pages: VecDeque<PageId>,
    /// A lookup table: "Text ID" -> "List of physical pages".
    mappings: BTreeMap<u64, BlockMapping>,
}

impl PagedAttentionConfig {
    /// Creates a new configuration for the memory manager.
    pub fn new(block_size: usize, total_pages: usize) -> Result<Self, PageManagerError> {
        if block_size == 0 {
            return Err(PageManagerError::InvalidBlockSize);
        }

        Ok(Self {
            block_size,
            page_tokens: block_size,
            total_pages,
            unified_memory: true,
            kv_type: KvCacheType::Fp32,
        })
    }

    /// Sets the numerical precision for the cache.
    pub fn with_kv_type(mut self, kv_type: KvCacheType) -> Self {
        self.kv_type = kv_type;
        self
    }
}

impl PagedAttentionBlockManager {
    /// Creates a new manager and fills it with empty pages.
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

    /// Returns the configuration.
    pub fn config(&self) -> &PagedAttentionConfig {
        &self.config
    }

    /// Finds the physical pages for a specific piece of text.
    pub fn mapped_pages(&self, handle: KvCacheHandle) -> Option<&[PageId]> {
        self.mappings.get(&handle.block_id).map(|mapping| mapping.pages.as_slice())
    }

    /// Calculates how many pages are needed to store a certain number of tokens.
    pub fn pages_for_tokens(&self, token_len: usize) -> usize {
        token_len.div_ceil(self.config.block_size)
    }

    /// ALLOCATE: Finds and reserves physical memory for a new request.
    pub fn allocate(
        &mut self,
        handle: KvCacheHandle,
        token_len: usize,
    ) -> Result<BlockMapping, PageManagerError> {
        let required_pages = self.pages_for_tokens(token_len);
        let available = self.free_pages.len();
        
        // Safety check: Do we have enough room?
        if required_pages > available {
            return Err(PageManagerError::OutOfPages {
                requested: required_pages,
                available,
            });
        }

        // Take pages from the "free" pile.
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
        // Remember the mapping.
        self.mappings.insert(handle.block_id, mapping.clone());
        Ok(mapping)
    }

    /// RELEASE: Takes memory back from a finished request and puts it in the "free" pile.
    pub fn release(&mut self, handle: KvCacheHandle) -> Result<(), PageManagerError> {
        let Some(mapping) = self.mappings.remove(&handle.block_id) else {
            return Err(PageManagerError::UnknownHandle(handle));
        };

        for page in mapping.pages {
            self.free_pages.push_back(page);
        }
        Ok(())
    }

    /// Returns the number of empty pages left.
    pub fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    /// Helper to convert a handle into a list of pages.
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
        // TEST: Renting spots in the garage
        // We start with 4 empty pages. We ask for memory to store 24 tokens.
        // Since each page holds 16 tokens, we need 2 pages.
        
        let config = PagedAttentionConfig::new(16, 4).unwrap();
        let mut manager = PagedAttentionBlockManager::new(config);
        let handle = KvCacheHandle {
            block_id: 1,
            token_len: 24,
        };

        let mapping = manager.allocate(handle, 24).unwrap();

        // Check if we got exactly 2 pages.
        assert_eq!(mapping.pages.len(), 2);
        // Check if there are 2 pages left for other people.
        assert_eq!(manager.free_pages(), 2);
        assert!(manager.mapped_pages(handle).is_some());
        assert_eq!(manager.materialize_span(handle).unwrap().pages.len(), 2);
    }

    #[test]
    fn release_returns_pages_to_pool() {
        // TEST: Returning spots to the garage
        // When we are done with a conversation, we should give the pages back
        // so someone else can use them.
        
        let config = PagedAttentionConfig::new(8, 2).unwrap();
        let mut manager = PagedAttentionBlockManager::new(config);
        let handle = KvCacheHandle {
            block_id: 2,
            token_len: 8,
        };

        // 1. Rent a spot.
        manager.allocate(handle, 8).unwrap();
        // 2. Return the spot.
        manager.release(handle).unwrap();

        // 3. The garage should be empty (2 free spots) again!
        assert_eq!(manager.free_pages(), 2);
    }

    #[test]
    fn rejects_oversized_allocations() {
        // TEST: The garage is full!
        // We only have 1 page (32 tokens), but we are asking for 64 tokens.
        // The manager should politely refuse.
        
        let config = PagedAttentionConfig::new(32, 1).unwrap();
        let mut manager = PagedAttentionBlockManager::new(config);
        let handle = KvCacheHandle {
            block_id: 3,
            token_len: 64,
        };

        let error = manager.allocate(handle, 64).unwrap_err();

        // It should return an 'OutOfPages' error.
        assert!(matches!(
            error,
            PageManagerError::OutOfPages {
                requested: 2,
                available: 1
            }
        ));
    }

    #[test]
    fn mapped_pages_miss() {
        let mgr = PagedAttentionBlockManager::new(PagedAttentionConfig::new(16, 32).unwrap());
        assert!(mgr.mapped_pages(KvCacheHandle { block_id: 1, token_len: 0 }).is_none());
    }

    #[test]
    fn release_error() {
        let mut mgr = PagedAttentionBlockManager::new(PagedAttentionConfig::new(16, 32).unwrap());
        assert!(mgr.release(KvCacheHandle { block_id: 1, token_len: 0 }).is_err());
    }

    #[test]
    fn trait_impls() {
        let span = PageSpan { pages: vec![PageId(1)] };
        let span2 = span.clone();
        assert_eq!(span, span2);
        assert!(format!("{:?}", span).contains("PageId"));
       
        let mapping = BlockMapping {
            handle: KvCacheHandle { block_id: 1, token_len: 1 },
            pages: vec![PageId(1)],
            token_len: 1,
        };
        let mapping2 = mapping.clone();
        assert_eq!(mapping, mapping2);
    }
}
