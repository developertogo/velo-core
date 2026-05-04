# Benchmarking Velo-Core

Velo-Core includes a comprehensive benchmarking harness to measure throughput, TTFT (Time To First Token), and cache performance.

## Smoke Benchmark
To run a quick performance sanity check:

```bash
cargo bench --bench smoke_bench
```

## Advanced Benchmarking
Use the `velo-bench` tool for detailed performance analysis:

```bash
cargo run --bin velo-bench -- --mode all --prompt-len 512 --gen-len 128
```

### Modes
- `prompt-processing`: Measures throughput for encoding the initial prompt.
- `generation`: Measures throughput for generating new tokens.
- `all`: Runs both modes.

### Comparison with llama.cpp
The `velo-bench` tool can consume `llama-bench` CSV output to provide direct speedup comparisons.
