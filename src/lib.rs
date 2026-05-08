//! Velo-Core: A high-performance inference engine for Apple Silicon.
//! 
//! This crate provides a fully optimized, AOT-compiled Metal backend for executing
//! large language models on macOS/iOS devices. It supports advanced features like
//! Paged Attention, Speculative Decoding (Tree Verification), and Multi-LoRA
//! dispatching, maximizing memory bandwidth utilization.

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
pub mod scheduler;
pub mod tokenizer;
pub mod sampling;
pub mod model_pool;
pub mod constraints;
pub mod ffi;
pub mod lora;
pub mod amx;
pub mod power;
pub mod nixl;
pub mod disagg;

pub use constraints::{CfgMatcher, Constraint};
pub use lora::{AdapterId, LoraRegistry, LoraConfig, LoraWeights};
pub use amx::AmxContext;
pub use power::{PowerTelemetry, SmcTelemetry, PrecisionGovernor, PowerState};
pub use nixl::{
    build_fabric, deserialize_block, serialize_block,
    CacheTransferAgent, DmaBlockManager, DmaStats, NixlNodeId, NodeRegistry,
    RemoteBlock, TransferOutcome, TransferStats, WIRE_HEADER_LEN,
};
pub use disagg::{
    build_disagg_pool, DecodeTask, DisaggPool, DisaggStats, NodeRole, PrefillTask,
};

pub use benchmark::{
    compare_with_llama_csv, load_llama_csv, parse_llama_csv, BenchmarkConfig, BenchmarkFormat,
    BenchmarkMode, BenchmarkReport, BenchmarkRow, BenchmarkSample, LlamaBenchRow,
    run_benchmark, run_single_case, HardwareSpecs,
};
pub use backend::{CausalLmBackend, GreedyDraftModel, GreedyTargetModel, TokenLogits};
pub use sampling::{GreedySampler, Sampler, TopPSampler, MinPSampler};
pub use model_pool::ModelPool;
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
pub use scheduler::VeloScheduler;
pub use gguf::{GgmlType, GgufError, GgufFile, GgufValue, TensorInfo};
pub use model_loader::{load_gguf, LoadError, ModelMeta, WeightStore};
pub use quant::{dequant_matrix, dequant_row};
pub use llama_cpu::LlamaCpuModel;
