//! Velo-Core: A high-performance inference engine for Apple Silicon.
//! 
//! This crate provides a fully optimized, AOT-compiled Metal backend for executing
//! large language models on macOS/iOS devices. It supports advanced features like
//! Paged Attention, Speculative Decoding (Tree Verification), and Multi-LoRA
//! dispatching, maximizing memory bandwidth utilization.
//!
//! ### Architecture Overview
//! Velo is designed as a disaggregated, hardware-aware inference stack. 
//! 1. **Engine layer** (`engine.rs`): Orchestrates memory and execution.
//! 2. **Scheduler layer** (`scheduler.rs`): Handles request admission and continuous batching.
//! 3. **Memory layer** (`paged_attention.rs`, `kv_store.rs`): Manages the KV-cache using paged memory.
//! 4. **Backend layer** (`metal/`): Executes kernels on the GPU.

/// Benchmark suite for measuring engine performance and roofline analysis.
pub mod benchmark;
/// Abstract traits for LLM backends (Draft and Target models).
pub mod backend;
/// Core orchestration logic for the inference engine.
pub mod engine;
/// GGUF file format parser for loading models.
pub mod gguf;
/// Key-Value store for persisting and retrieving cached sequences.
pub mod kv_store;
/// Reference CPU implementation of the Llama architecture.
pub mod llama_cpu;
/// Apple Metal GPU acceleration backend and MSL kernels.
pub mod metal;
/// Mock backend for testing engine logic without a GPU.
pub mod mock_backend;
/// Utilities for loading weights and mapping them to memory.
pub mod model_loader;
/// Paged Attention implementation for efficient memory management.
pub mod paged_attention;
/// Low-level quantization and dequantization routines (e.g. Q4_K, INT8).
pub mod quant;
/// Radix-tree based prefix cache for reusing KV-blocks across requests.
pub mod radix_cache;
/// Memory management and execution context abstraction.
pub mod runtime;
/// Manages stable identifiers (slots) for concurrent requests.
pub mod slot_manager;
/// Speculative decoding (Draft-then-Verify) implementation.
pub mod speculative;
/// High-level asynchronous scheduler for continuous batching.
pub mod scheduler;
/// Tokenization and vocabulary mapping.
pub mod tokenizer;
/// Advanced sampling strategies (Top-P, Min-P, Mirostat).
pub mod sampling;
/// Pool for managing multiple models in memory.
pub mod model_pool;
/// Structured output constraints (Grammar, Regex, JSON).
pub mod constraints;
/// Foreign Function Interface for C/C++ integration.
pub mod ffi;
/// Low-Rank Adaptation (LoRA) weight management and dispatch.
pub mod lora;
/// Apple Matrix Co-processor (AMX) integration for CPU acceleration.
pub mod amx;
/// Power and thermal telemetry monitoring.
pub mod power;
/// NIXL fabric for high-speed inter-node KV-cache migration.
pub mod nixl;
/// Disaggregated serving orchestration (Prefill/Decode split).
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
