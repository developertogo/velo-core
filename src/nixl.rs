//! NIXL-Inspired KV-Cache Transfer
//!
//! Implements the three-layer cache migration stack modelled after NVIDIA's NIXL
//! (Network Infrastructure for eXpress-fabric Learning) design:
//!
//! 1. **DMA Block Manager** — serializes/deserializes KV-cache blocks into a
//!    portable wire format with zero extra allocations on the hot path.
//! 2. **Node Registry** — lightweight in-process peer-discovery fabric that
//!    simulates a multi-node memory mesh on a single Apple Silicon host.
//! 3. **Cache Transfer Agent** — orchestrates the eviction-handoff loop:
//!    when local GPU memory overflows, cold Radix Tree branches are streamed
//!    to the least-loaded peer instead of being dropped to disk.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::radix_cache::KvCacheHandle;

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

/// Portable, length-prefixed serialisation of a single KV-cache block.
///
/// Layout (little-endian):
/// ```text
/// [block_id: u64][token_len: u32][byte_len: u32][payload: byte_len bytes]
/// ```
pub const WIRE_HEADER_LEN: usize = 16; // 8 + 4 + 4

/// Serialise a KV-block handle + its raw bytes into the NIXL wire format.
///
/// In production this would be a scatter-gather DMA into a registered memory
/// region; here we memcpy into a `Vec<u8>` that the caller owns.
pub fn serialize_block(handle: KvCacheHandle, payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(WIRE_HEADER_LEN + payload.len());
    buf.extend_from_slice(&handle.block_id.to_le_bytes());
    buf.extend_from_slice(&(handle.token_len as u32).to_le_bytes());
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(payload);
    buf
}

/// Deserialise a NIXL wire buffer produced by [`serialize_block`].
///
/// Returns `(handle, payload_slice)` on success, or `None` if the buffer is
/// malformed or too short.
pub fn deserialize_block(buf: &[u8]) -> Option<(KvCacheHandle, &[u8])> {
    if buf.len() < WIRE_HEADER_LEN {
        return None;
    }
    let block_id = u64::from_le_bytes(buf[0..8].try_into().ok()?);
    let token_len = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;
    let byte_len = u32::from_le_bytes(buf[12..16].try_into().ok()?) as usize;
    if buf.len() < WIRE_HEADER_LEN + byte_len {
        return None;
    }
    let handle = KvCacheHandle { block_id, token_len };
    Some((handle, &buf[WIRE_HEADER_LEN..WIRE_HEADER_LEN + byte_len]))
}

// ---------------------------------------------------------------------------
// DMA Block Manager
// ---------------------------------------------------------------------------

/// Manages a pool of registered memory regions for zero-copy block transfer.
///
/// Each region is a contiguous byte slice that can hold exactly one KV-block.
/// On real hardware this would map to `MTLBuffer` storage mode `.storageModeShared`
/// so the CPU and GPU share the same physical pages — no PCIe round-trip needed.
#[derive(Debug)]
pub struct DmaBlockManager {
    /// bytes_per_token × max_tokens_per_block
    block_bytes: usize,
    /// Available (free) region indices.
    free_pool: Vec<usize>,
    /// Backing store: one flat Vec per registered region.
    regions: Vec<Vec<u8>>,
    /// Transfer statistics.
    stats: DmaStats,
}

/// Telemetry counters for the DMA subsystem.
#[derive(Debug, Clone, Default)]
pub struct DmaStats {
    pub regions_allocated: usize,
    pub blocks_serialized: usize,
    pub blocks_deserialized: usize,
    pub bytes_transferred: u64,
}

impl DmaBlockManager {
    /// Create a new manager pre-populating `pool_size` registered regions.
    pub fn new(block_bytes: usize, pool_size: usize) -> Self {
        let regions: Vec<Vec<u8>> = (0..pool_size).map(|_| vec![0u8; block_bytes]).collect();
        let free_pool: Vec<usize> = (0..pool_size).collect();
        Self {
            block_bytes,
            free_pool,
            regions,
            stats: DmaStats {
                regions_allocated: pool_size,
                ..Default::default()
            },
        }
    }

    /// Acquire a free region index, growing the pool if necessary.
    pub fn acquire(&mut self) -> usize {
        if self.free_pool.is_empty() {
            let idx = self.regions.len();
            self.regions.push(vec![0u8; self.block_bytes]);
            self.stats.regions_allocated += 1;
            return idx;
        }
        self.free_pool.pop().unwrap()
    }

    /// Release a region back to the free pool.
    pub fn release(&mut self, idx: usize) {
        debug_assert!(idx < self.regions.len(), "invalid region index");
        self.free_pool.push(idx);
    }

    /// Write `payload` into region `idx` and return the NIXL wire buffer.
    ///
    /// If `payload` is shorter than `block_bytes`, the remainder is zeroed.
    pub fn write_and_serialize(
        &mut self,
        idx: usize,
        handle: KvCacheHandle,
        payload: &[u8],
    ) -> Vec<u8> {
        let region = &mut self.regions[idx];
        let copy_len = payload.len().min(self.block_bytes);
        region[..copy_len].copy_from_slice(&payload[..copy_len]);
        if copy_len < self.block_bytes {
            region[copy_len..].fill(0);
        }
        self.stats.blocks_serialized += 1;
        self.stats.bytes_transferred += payload.len() as u64;
        serialize_block(handle, &region[..copy_len])
    }

    /// Deserialise a wire buffer into region `idx` and return `(handle, len)`.
    pub fn receive_into(&mut self, idx: usize, wire: &[u8]) -> Option<(KvCacheHandle, usize)> {
        let (handle, payload) = deserialize_block(wire)?;
        let region = &mut self.regions[idx];
        let copy_len = payload.len().min(self.block_bytes);
        region[..copy_len].copy_from_slice(&payload[..copy_len]);
        self.stats.blocks_deserialized += 1;
        self.stats.bytes_transferred += copy_len as u64;
        Some((handle, copy_len))
    }

    /// Read the raw bytes of region `idx`.
    pub fn region(&self, idx: usize) -> &[u8] {
        &self.regions[idx]
    }

    pub fn stats(&self) -> &DmaStats {
        &self.stats
    }

    pub fn pool_size(&self) -> usize {
        self.regions.len()
    }

    pub fn free_regions(&self) -> usize {
        self.free_pool.len()
    }
}

// ---------------------------------------------------------------------------
// Node Registry (peer discovery)
// ---------------------------------------------------------------------------

/// Unique identifier for a memory node in the fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NixlNodeId(pub u32);

impl std::fmt::Display for NixlNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node-{}", self.0)
    }
}

/// Snapshot of a peer's current memory pressure.
#[derive(Debug, Clone)]
pub struct PeerStatus {
    pub node_id: NixlNodeId,
    /// Fraction of capacity currently in use [0.0, 1.0].
    pub utilisation: f32,
    /// Tokens of free capacity this peer can accept.
    pub free_tokens: usize,
    /// Approximate RTT to this peer (simulated).
    pub rtt: Duration,
}

/// Shared in-process registry — all nodes register here on startup.
#[derive(Debug, Default)]
pub struct NodeRegistry {
    inner: Mutex<RegistryInner>,
}

#[derive(Debug, Default)]
struct RegistryInner {
    next_id: u32,
    peers: HashMap<NixlNodeId, PeerStatus>,
}

impl NodeRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Register a new node and return its assigned ID.
    pub fn register(&self, capacity_tokens: usize) -> NixlNodeId {
        let mut inner = self.inner.lock().unwrap();
        let id = NixlNodeId(inner.next_id);
        inner.next_id += 1;
        inner.peers.insert(
            id,
            PeerStatus {
                node_id: id,
                utilisation: 0.0,
                free_tokens: capacity_tokens,
                rtt: Duration::from_micros(50 + (id.0 as u64 * 10)),
            },
        );
        id
    }

    /// Update this node's current memory utilisation.
    pub fn heartbeat(&self, id: NixlNodeId, utilisation: f32, free_tokens: usize) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(peer) = inner.peers.get_mut(&id) {
            peer.utilisation = utilisation;
            peer.free_tokens = free_tokens;
        }
    }

    /// Return a snapshot of all peers except `exclude`.
    pub fn peers_except(&self, exclude: NixlNodeId) -> Vec<PeerStatus> {
        let inner = self.inner.lock().unwrap();
        inner
            .peers
            .values()
            .filter(|p| p.node_id != exclude)
            .cloned()
            .collect()
    }

    /// Select the least-loaded peer that can accept `needed_tokens`.
    pub fn best_peer(&self, exclude: NixlNodeId, needed_tokens: usize) -> Option<PeerStatus> {
        self.peers_except(exclude)
            .into_iter()
            .filter(|p| p.free_tokens >= needed_tokens)
            .min_by(|a, b| a.utilisation.partial_cmp(&b.utilisation).unwrap())
    }

    pub fn node_count(&self) -> usize {
        self.inner.lock().unwrap().peers.len()
    }
}

// ---------------------------------------------------------------------------
// Cache Transfer Agent
// ---------------------------------------------------------------------------

/// A transferred block sitting in a peer's remote inbox.
#[derive(Debug, Clone)]
pub struct RemoteBlock {
    pub handle: KvCacheHandle,
    pub origin: NixlNodeId,
    pub received_at: Instant,
    /// The raw payload bytes (simulates shared-memory or RDMA buffer).
    pub payload: Vec<u8>,
}

/// Per-transfer-attempt outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferOutcome {
    /// Block shipped to `peer`.
    Sent { peer: NixlNodeId, bytes: usize },
    /// No peer available; block was dropped locally.
    DroppedLocally,
    /// Peer rejected the block (out of capacity).
    PeerFull { peer: NixlNodeId },
}

/// Telemetry for the transfer agent.
#[derive(Debug, Clone, Default)]
pub struct TransferStats {
    pub blocks_sent: u64,
    pub blocks_dropped: u64,
    pub bytes_sent: u64,
    pub transfers_failed: u64,
}

/// Orchestrates the eviction-handoff loop.
///
/// When the local [`RadixCache`] overflows, [`CacheTransferAgent::handoff`] is
/// called with a list of cold [`KvCacheHandle`]s.  For each handle it:
/// 1. Acquires a DMA region.
/// 2. Serialises the block payload.
/// 3. Selects the best peer via the [`NodeRegistry`].
/// 4. "Ships" the wire buffer (in-process: pushes into the peer's inbox).
/// 5. Returns [`TransferOutcome`]s for observability.
pub struct CacheTransferAgent {
    pub node_id: NixlNodeId,
    registry: Arc<NodeRegistry>,
    dma: DmaBlockManager,
    /// Simulated receive inbox: maps peer NodeId → list of received blocks.
    inboxes: Arc<Mutex<HashMap<NixlNodeId, Vec<RemoteBlock>>>>,
    pub stats: TransferStats,
}

impl std::fmt::Debug for CacheTransferAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheTransferAgent")
            .field("node_id", &self.node_id)
            .field("stats", &self.stats)
            .finish()
    }
}

impl CacheTransferAgent {
    /// Create a new agent and register it with the fabric.
    pub fn new(
        registry: Arc<NodeRegistry>,
        inboxes: Arc<Mutex<HashMap<NixlNodeId, Vec<RemoteBlock>>>>,
        block_bytes: usize,
        dma_pool_size: usize,
        node_capacity_tokens: usize,
    ) -> Self {
        let node_id = registry.register(node_capacity_tokens);
        // Ensure this node has an inbox slot.
        inboxes.lock().unwrap().entry(node_id).or_default();
        Self {
            node_id,
            registry,
            dma: DmaBlockManager::new(block_bytes, dma_pool_size),
            inboxes,
            stats: TransferStats::default(),
        }
    }

    /// Attempt to hand off a set of evicted blocks to the best available peer.
    ///
    /// `payloads` must be parallel to `handles`: `payloads[i]` is the raw KV
    /// bytes for `handles[i]`.  Pass an empty slice if you only have handles
    /// (the wire payload will be all-zeros, which is valid for testing).
    pub fn handoff(
        &mut self,
        handles: &[KvCacheHandle],
        payloads: &[Vec<u8>],
    ) -> Vec<TransferOutcome> {
        let empty: Vec<u8> = vec![];
        let mut outcomes = Vec::with_capacity(handles.len());

        for (i, &handle) in handles.iter().enumerate() {
            let payload = payloads.get(i).unwrap_or(&empty);
            let needed = handle.token_len;

            let Some(peer) = self.registry.best_peer(self.node_id, needed) else {
                self.stats.blocks_dropped += 1;
                outcomes.push(TransferOutcome::DroppedLocally);
                continue;
            };

            let region_idx = self.dma.acquire();
            let wire = self.dma.write_and_serialize(region_idx, handle, payload);
            let bytes = wire.len();
            self.dma.release(region_idx);

            // Deserialise on the receiving side (simulates RDMA write + doorbell).
            let recv_idx = {
                // We need a temporary DmaBlockManager on the peer side — in a real
                // system the peer owns its own DMA manager.  Here we re-use ours
                // for the deserialization parse only.
                let recv_region = self.dma.acquire();
                let result = self.dma.receive_into(recv_region, &wire);
                self.dma.release(recv_region);
                result
            };

            if recv_idx.is_none() {
                self.stats.transfers_failed += 1;
                outcomes.push(TransferOutcome::PeerFull { peer: peer.node_id });
                continue;
            }

            // Deliver to inbox.
            let remote = RemoteBlock {
                handle,
                origin: self.node_id,
                received_at: Instant::now(),
                payload: wire[WIRE_HEADER_LEN..].to_vec(),
            };
            self.inboxes
                .lock()
                .unwrap()
                .entry(peer.node_id)
                .or_default()
                .push(remote);

            self.stats.blocks_sent += 1;
            self.stats.bytes_sent += bytes as u64;
            outcomes.push(TransferOutcome::Sent {
                peer: peer.node_id,
                bytes,
            });
        }

        outcomes
    }

    /// Drain all blocks delivered to this node's inbox.
    pub fn drain_inbox(&self) -> Vec<RemoteBlock> {
        self.inboxes
            .lock()
            .unwrap()
            .get_mut(&self.node_id)
            .map(|v| std::mem::take(v))
            .unwrap_or_default()
    }

    pub fn dma_stats(&self) -> &DmaStats {
        self.dma.stats()
    }
}

/// Convenience builder — creates the shared registry and inbox map, then
/// returns `count` agents wired together.
pub fn build_fabric(
    count: usize,
    block_bytes: usize,
    dma_pool: usize,
    capacity_tokens: usize,
) -> (Arc<NodeRegistry>, Vec<CacheTransferAgent>) {
    let registry = NodeRegistry::new();
    let inboxes: Arc<Mutex<HashMap<NixlNodeId, Vec<RemoteBlock>>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let agents = (0..count)
        .map(|_| {
            CacheTransferAgent::new(
                Arc::clone(&registry),
                Arc::clone(&inboxes),
                block_bytes,
                dma_pool,
                capacity_tokens,
            )
        })
        .collect();

    (registry, agents)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_handle(block_id: u64, token_len: usize) -> KvCacheHandle {
        KvCacheHandle { block_id, token_len }
    }

    // --- Wire format ---

    #[test]
    fn round_trip_serialize_deserialize() {
        let handle = dummy_handle(42, 8);
        let payload = vec![0xABu8; 64];
        let wire = serialize_block(handle, &payload);

        assert_eq!(wire.len(), WIRE_HEADER_LEN + 64);
        let (out_handle, out_payload) = deserialize_block(&wire).unwrap();
        assert_eq!(out_handle, handle);
        assert_eq!(out_payload, payload.as_slice());
    }

    #[test]
    fn deserialize_rejects_short_header() {
        assert!(deserialize_block(&[0u8; WIRE_HEADER_LEN - 1]).is_none());
    }

    #[test]
    fn deserialize_rejects_truncated_payload() {
        let handle = dummy_handle(1, 4);
        let payload = vec![0u8; 32];
        let mut wire = serialize_block(handle, &payload);
        wire.truncate(wire.len() - 1); // corrupt: cut last byte
        assert!(deserialize_block(&wire).is_none());
    }

    #[test]
    fn serialize_empty_payload() {
        let handle = dummy_handle(7, 0);
        let wire = serialize_block(handle, &[]);
        assert_eq!(wire.len(), WIRE_HEADER_LEN);
        let (h, p) = deserialize_block(&wire).unwrap();
        assert_eq!(h, handle);
        assert!(p.is_empty());
    }

    // --- DmaBlockManager ---

    #[test]
    fn dma_write_and_read_back() {
        let mut mgr = DmaBlockManager::new(128, 4);
        assert_eq!(mgr.pool_size(), 4);
        assert_eq!(mgr.free_regions(), 4);

        let handle = dummy_handle(1, 4);
        let payload = vec![0x55u8; 64];
        let idx = mgr.acquire();
        let wire = mgr.write_and_serialize(idx, handle, &payload);
        mgr.release(idx);

        assert_eq!(mgr.stats().blocks_serialized, 1);
        assert_eq!(mgr.stats().bytes_transferred, 64);

        let recv_idx = mgr.acquire();
        let (h, len) = mgr.receive_into(recv_idx, &wire).unwrap();
        assert_eq!(h, handle);
        assert_eq!(len, 64);
        assert_eq!(&mgr.region(recv_idx)[..64], &[0x55u8; 64]);
        mgr.release(recv_idx);
    }

    #[test]
    fn dma_pool_grows_on_demand() {
        let mut mgr = DmaBlockManager::new(64, 1);
        let _a = mgr.acquire();
        let _b = mgr.acquire(); // should trigger growth
        assert!(mgr.pool_size() >= 2);
    }

    #[test]
    fn dma_receive_rejects_bad_wire() {
        let mut mgr = DmaBlockManager::new(64, 2);
        let idx = mgr.acquire();
        assert!(mgr.receive_into(idx, &[0u8; 4]).is_none());
        mgr.release(idx);
    }

    // --- NodeRegistry ---

    #[test]
    fn registry_registers_and_discovers_peers() {
        let registry = NodeRegistry::new();
        let a = registry.register(1000);
        let b = registry.register(2000);
        let c = registry.register(500);

        assert_eq!(registry.node_count(), 3);

        let peers = registry.peers_except(a);
        assert_eq!(peers.len(), 2);
        assert!(peers.iter().any(|p| p.node_id == b));
        assert!(peers.iter().any(|p| p.node_id == c));
    }

    #[test]
    fn registry_selects_least_loaded_peer() {
        let registry = NodeRegistry::new();
        let a = registry.register(1000);
        let b = registry.register(1000);
        let c = registry.register(1000);

        registry.heartbeat(b, 0.8, 200);
        registry.heartbeat(c, 0.2, 800);

        let best = registry.best_peer(a, 100).unwrap();
        assert_eq!(best.node_id, c);
    }

    #[test]
    fn registry_returns_none_when_all_full() {
        let registry = NodeRegistry::new();
        let a = registry.register(10);
        let b = registry.register(10);
        registry.heartbeat(b, 1.0, 0);

        assert!(registry.best_peer(a, 100).is_none());
    }

    // --- CacheTransferAgent ---

    #[test]
    fn handoff_sends_blocks_to_peer() {
        let (registry, mut agents) = build_fabric(2, 256, 4, 10_000);
        assert_eq!(registry.node_count(), 2);

        let sender = &mut agents[0];
        let handles = vec![dummy_handle(1, 8), dummy_handle(2, 4)];
        let payloads = vec![vec![0xAAu8; 64], vec![0xBBu8; 32]];

        let outcomes = sender.handoff(&handles, &payloads);
        assert_eq!(outcomes.len(), 2);
        for outcome in &outcomes {
            assert!(matches!(outcome, TransferOutcome::Sent { .. }));
        }
        assert_eq!(sender.stats.blocks_sent, 2);
        assert!(sender.stats.bytes_sent > 0);
    }

    #[test]
    fn handoff_drops_when_no_peer_available() {
        let (_, mut agents) = build_fabric(1, 256, 2, 1000);
        let handles = vec![dummy_handle(99, 5)];
        let outcomes = agents[0].handoff(&handles, &[]);

        assert_eq!(outcomes, vec![TransferOutcome::DroppedLocally]);
        assert_eq!(agents[0].stats.blocks_dropped, 1);
    }

    #[test]
    fn receiver_drains_inbox() {
        let (_, mut agents) = build_fabric(2, 256, 4, 10_000);

        // Agent 0 ships a block to Agent 1's inbox.
        let handles = vec![dummy_handle(7, 4)];
        let payloads = vec![vec![0xCCu8; 32]];
        agents[0].handoff(&handles, &payloads);

        let received = agents[1].drain_inbox();
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].handle, dummy_handle(7, 4));
        assert_eq!(received[0].origin, agents[0].node_id);

        // Second drain should be empty.
        assert!(agents[1].drain_inbox().is_empty());
    }

    #[test]
    fn handoff_without_payloads_uses_zero_bytes() {
        let (_, mut agents) = build_fabric(2, 64, 2, 5000);
        let handles = vec![dummy_handle(3, 2)];
        let outcomes = agents[0].handoff(&handles, &[]); // no payload slice
        assert!(matches!(outcomes[0], TransferOutcome::Sent { .. }));
    }

    #[test]
    fn dma_stats_accumulate_across_handoffs() {
        let (_, mut agents) = build_fabric(2, 128, 4, 10_000);
        let handles = vec![
            dummy_handle(10, 4),
            dummy_handle(11, 4),
            dummy_handle(12, 4),
        ];
        agents[0].handoff(&handles, &[]);
        assert_eq!(agents[0].dma_stats().blocks_serialized, 3);
    }

    #[test]
    fn nixl_node_id_display() {
        let id = NixlNodeId(5);
        assert_eq!(format!("{id}"), "node-5");
    }

    #[test]
    fn remote_block_carries_origin_and_payload() {
        let (_, mut agents) = build_fabric(3, 256, 4, 10_000);
        let payload = vec![0xDEu8; 100];
        agents[0].handoff(&[dummy_handle(1, 8)], &[payload.clone()]);

        // Drain either agent 1 or agent 2 — one should have it.
        let mut received = agents[1].drain_inbox();
        received.extend(agents[2].drain_inbox());
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].origin, agents[0].node_id);
        assert!(!received[0].payload.is_empty());
    }
}
