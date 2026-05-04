pub mod benchmark;
pub mod backend;
pub mod engine;
pub mod gguf;
pub mod kv_store;
pub mod llama_cpu;
pub mod metal;
pub mod mock_backend;
pub mod model_loader;
pub mod paged_attention;
pub mod quant;
pub mod radix_cache;
pub mod runtime;
pub mod slot_manager;
pub mod speculative;

pub use benchmark::{
    compare_with_llama_csv, load_llama_csv, parse_llama_csv, BenchmarkConfig, BenchmarkFormat,
    BenchmarkMode, BenchmarkReport, BenchmarkRow, BenchmarkSample, LlamaBenchRow,
    run_benchmark, run_single_case,
};
pub use backend::{CausalLmBackend, GreedyDraftModel, GreedySampler, GreedyTargetModel, TokenLogits};
pub use engine::{
    EngineConfig, EngineError, EngineOutput, EnginePrefillOutput, EngineStats, VeloEngine,
};
pub use kv_store::{InMemoryKvStore, KvBlock, KvStore, KvStoreError};
pub use metal::{
    MetalBackend, MetalBackendConfig, MetalBufferPlacement, MetalDeviceInfo,
    MetalMemoryRuntime, MetalRuntimeConfig, MetalRuntimeContext, MetalRuntimeHandles,
};
pub use mock_backend::MockBackend;
pub use paged_attention::{
    BlockMapping, PageId, PageManagerError, PageSpan, PagedAttentionBlockManager,
    PagedAttentionConfig,
};
pub use radix_cache::{CacheInsert, CacheLookup, KvCacheHandle, RadixCache, TokenId};
pub use runtime::{
    CpuMemoryRuntime, KvBlockStore, MemoryRuntime, MemoryRuntimeConfig, PagedBlockAllocator,
};
pub use slot_manager::{SlotId, SlotPool};
pub use speculative::{
    DraftModel, NextTokenPrediction, SpeculativeDecoder, SpeculativeError, SpeculativeOutput,
    SpeculativeStats, TargetModel, VerifyStep, Result,
};
pub use gguf::{GgmlType, GgufError, GgufFile, GgufValue, TensorInfo};
pub use model_loader::{load_gguf, LoadError, ModelMeta, WeightStore};
pub use quant::{dequant_matrix, dequant_row};
pub use llama_cpu::LlamaCpuModel;
