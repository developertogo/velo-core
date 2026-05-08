use std::sync::{Arc, Mutex};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions};

use crate::paged_attention::{BlockMapping, PageManagerError, PageSpan, PagedAttentionBlockManager, PagedAttentionConfig};
use crate::radix_cache::{CacheLookup, KvCacheHandle};
use crate::runtime::{MemoryRuntime, MemoryRuntimeConfig, PagedBlockAllocator};
use crate::speculative::{Result, SpeculativeError};

use super::config::MetalRuntimeConfig;
use super::kv_store::SharedMetalKvStore;
use super::types::{MetalBufferPlacement, MetalDeviceInfo};

// ── AOT / JIT Library Loading ─────────────────────────────────────────────────

/// Loads the Metal shader library, preferring an AOT pre-compiled `.metallib`
/// over runtime JIT source compilation.
///
/// ## AOT Path (preferred)
/// If `build.rs` successfully compiled `src/kernels.metal` into a `.metallib`
/// at build time, its path is embedded via the `VELO_METALLIB_PATH` env var.
/// We load it with `newLibraryWithURL_error` — zero shader compilation at
/// startup, eliminating the 200–800 ms JIT penalty on cold inference.
///
/// ## JIT Fallback
/// When the AOT artifact is absent (CI without Xcode, `VELO_SKIP_METALLIB=1`,
/// cross-compilation), falls back to `newLibraryWithSource_options_error`.
fn load_metal_library(
    device: &ProtocolObject<dyn MTLDevice>,
    kernel_source: &str,
) -> std::result::Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    // ── AOT path: load via file URL ────────────────────────────────────────
    if let Some(metallib_path) = option_env!("VELO_METALLIB_PATH") {
        let url_str = format!("file://{}", metallib_path);
        if let Some(url) = objc2_foundation::NSURL::URLWithString(
            &objc2_foundation::NSString::from_str(&url_str),
        ) {
            match device.newLibraryWithURL_error(&url) {
                Ok(lib) => {
                    return Ok(lib);
                }
                Err(e) => {
                    eprintln!(
                        "[velo-core] Warning: AOT .metallib load failed ({:?}); falling back to JIT.",
                        e
                    );
                }
            }
        }
    }

    // ── JIT fallback: compile MSL source at runtime ────────────────────────
    device
        .newLibraryWithSource_options_error(
            &objc2_foundation::NSString::from_str(kernel_source),
            None,
        )
        .map_err(|e| format!("JIT Metal compilation failed: {:?}", e))
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

unsafe impl Send for MetalRuntimeHandles {}
unsafe impl Sync for MetalRuntimeHandles {}

impl Clone for MetalRuntimeHandles {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            command_queue: self.command_queue.clone(),
            library: self.library.clone(),
        }
    }
}

/// Runtime context containing hardware info and handles.
pub struct MetalRuntimeContext {
    pub device: MetalDeviceInfo,
    pub memory: MemoryRuntimeConfig,
    pub placement: MetalBufferPlacement,
    pub handles: MetalRuntimeHandles,
}

unsafe impl Send for MetalRuntimeContext {}
unsafe impl Sync for MetalRuntimeContext {}

#[derive(Clone)]
pub struct SharedPagedAttentionBlockManager(pub Arc<Mutex<PagedAttentionBlockManager>>);

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

impl std::fmt::Debug for SharedPagedAttentionBlockManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedPagedAttentionBlockManager")
    }
}

/// A memory runtime that manages Metal-backed KV cache and paged attention.
pub struct MetalMemoryRuntime {
    pub config: MetalRuntimeConfig,
    pub context: MetalRuntimeContext,
    pub store: SharedMetalKvStore,
    pub allocator: SharedPagedAttentionBlockManager,
    pub bound_prefix: Option<CacheLookup>,
    /// Persistent GPU buffer mapping SlotId -> [PageId; MaxPages].
    pub slot_mapping: Retained<ProtocolObject<dyn MTLBuffer>>,
}

unsafe impl Send for MetalMemoryRuntime {}
unsafe impl Sync for MetalMemoryRuntime {}

impl MetalMemoryRuntime {
    /// Creates a new MetalMemoryRuntime from the provided configuration.
    /// This initializes the Metal device, command queue, and shader library.
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

        let kernel_source = include_str!("../kernels.metal");
        let library = load_metal_library(&device, kernel_source)
            .map_err(|e| SpeculativeError::Model(format!("Failed to load Metal kernels: {}", e)))?;

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
        let store = SharedMetalKvStore(Arc::new(Mutex::new(crate::metal::kv_store::MetalKvStore::new(
            device.clone(),
            config.memory.bytes_per_token,
            config.memory.paged_total_pages,
            config.memory.paged_block_size,
            config.memory.n_layer,
        ))));
       
        let allocator = SharedPagedAttentionBlockManager(Arc::new(Mutex::new(PagedAttentionBlockManager::new(
            PagedAttentionConfig::new(
                config.memory.paged_block_size,
                config.memory.paged_total_pages,
            )
            .map_err(|error| SpeculativeError::Model(error.to_string()))?,
        ))));

        // Allocate slot mapping buffer
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

    /// Injects specific Metal hardware handles into the runtime context.
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

    fn metal_handles(&self) -> Option<crate::metal::runtime::MetalRuntimeHandles> {
        Some(self.context.handles.clone())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::MetalRuntimeConfig;

    #[test]
    fn test_runtime_validation() {
        let mut cfg = MetalRuntimeConfig::default();
        cfg.model_name = "".into();
        assert!(MetalMemoryRuntime::new(cfg).is_err());

        let mut cfg = MetalRuntimeConfig::default();
        cfg.model_name = "test".into();
        cfg.memory.bytes_per_token = 0;
        assert!(MetalMemoryRuntime::new(cfg).is_err());

        let mut cfg = MetalRuntimeConfig::default();
        cfg.model_name = "test".into();
        cfg.memory.paged_block_size = 0;
        assert!(MetalMemoryRuntime::new(cfg).is_err());
    }

    #[test]
    fn test_runtime_handles_clone() {
        let handles = MetalRuntimeHandles {
            device: None,
            command_queue: None,
            library: None,
        };
        let handles2 = handles.clone();
        assert!(handles2.device.is_none());
    }

    #[test]
    fn test_metal_runtime_context_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MetalRuntimeContext>();
        assert_send_sync::<MetalMemoryRuntime>();
    }
    
    #[test]
    fn test_shared_paged_attention_block_manager_debug() {
        let mgr = SharedPagedAttentionBlockManager(Arc::new(Mutex::new(crate::paged_attention::PagedAttentionBlockManager::new(crate::paged_attention::PagedAttentionConfig {
            total_pages: 10,
            block_size: 16,
            page_tokens: 16,
            kv_type: crate::paged_attention::KvCacheType::Fp32,
            unified_memory: true,
        }))));
        assert!(format!("{:?}", mgr).contains("SharedPagedAttentionBlockManager"));
    }
}
