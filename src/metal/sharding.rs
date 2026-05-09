//! Weight sharding utilities for Tensor Parallelism.

use crate::metal::Quantization;
use crate::speculative::{Result, SpeculativeError};

/// Type of sharding to apply to a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// No sharding (replicated across all devices).
    Replicated,
    /// Column-parallel: split along the output dimension (rows in the matrix).
    Column,
    /// Row-parallel: split along the input dimension (columns in the matrix).
    Row,
}

/// A sharded view of a weight tensor.
pub struct ShardedWeight {
    pub data: Vec<u8>,
    pub shape: Vec<u64>,
}

/// Shards a weight tensor based on the specified strategy.
///
/// GGUF tensors are usually [cols, rows].
/// - Column-parallel splits 'rows'.
/// - Row-parallel splits 'cols'.
pub fn shard_weight(
    name: &str,
    data: &[u8],
    shape: &[u64],
    quant: Quantization,
    strategy: ShardStrategy,
    tp_degree: usize,
    tp_rank: usize,
) -> Result<ShardedWeight> {
    if tp_degree <= 1 || strategy == ShardStrategy::Replicated {
        return Ok(ShardedWeight {
            data: data.to_vec(),
            shape: shape.to_vec(),
        });
    }

    match strategy {
        ShardStrategy::Column => {
            // Split along the last dimension (rows).
            let rows = shape[shape.len() - 1] as usize;
            if rows % tp_degree != 0 {
                return Err(SpeculativeError::Model(format!(
                    "Tensor {} rows ({}) not divisible by TP degree {}",
                    name, rows, tp_degree
                )));
            }
            let shard_rows = rows / tp_degree;
            let bytes_per_row = data.len() / rows;
            let start = tp_rank * shard_rows * bytes_per_row;
            let end = (tp_rank + 1) * shard_rows * bytes_per_row;

            let mut shard_shape = shape.to_vec();
            let last_idx = shard_shape.len() - 1;
            shard_shape[last_idx] = shard_rows as u64;

            Ok(ShardedWeight {
                data: data[start..end].to_vec(),
                shape: shard_shape,
            })
        }
        ShardStrategy::Row => {
            // Split along the first dimension (cols).
            // For quantized weights, we must respect block boundaries.
            let cols = shape[0] as usize;
            if cols % tp_degree != 0 {
                return Err(SpeculativeError::Model(format!(
                    "Tensor {} cols ({}) not divisible by TP degree {}",
                    name, cols, tp_degree
                )));
            }
            let shard_cols = cols / tp_degree;
            let block_size = quant.block_size();
            if shard_cols % block_size != 0 {
                return Err(SpeculativeError::Model(format!(
                    "Tensor {} shard cols ({}) must be multiple of block size {}",
                    name, shard_cols, block_size
                )));
            }

            let rows = if shape.len() > 1 { shape[1] as usize } else { 1 };
            let bytes_per_row = data.len() / rows;
            let bytes_per_shard_col_row = bytes_per_row / tp_degree;
            
            let mut shard_data = Vec::with_capacity(shard_cols * rows * (bytes_per_row / cols));
            
            for r in 0..rows {
                let row_start = r * bytes_per_row;
                let shard_start = row_start + tp_rank * bytes_per_shard_col_row;
                let shard_end = shard_start + bytes_per_shard_col_row;
                shard_data.extend_from_slice(&data[shard_start..shard_end]);
            }

            let mut shard_shape = shape.to_vec();
            shard_shape[0] = shard_cols as u64;

            Ok(ShardedWeight {
                data: shard_data,
                shape: shard_shape,
            })
        }
        ShardStrategy::Replicated => unreachable!(),
    }
}

pub fn get_shard_strategy(name: &str) -> ShardStrategy {
    if name.contains("attention.wq") || name.contains("attention.wk") || name.contains("attention.wv") {
        ShardStrategy::Column
    } else if name.contains("attention.wo") {
        ShardStrategy::Row
    } else if name.contains("feed_forward.w1") || name.contains("feed_forward.w3") {
        ShardStrategy::Column
    } else if name.contains("feed_forward.w2") {
        ShardStrategy::Row
    } else if name == "output.weight" {
        ShardStrategy::Column
    } else {
        ShardStrategy::Replicated
    }
}
