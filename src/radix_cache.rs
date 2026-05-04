use std::collections::BTreeMap;

pub type TokenId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvCacheHandle {
    pub block_id: u64,
    pub token_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheLookup {
    pub matched_tokens: usize,
    pub handle: Option<KvCacheHandle>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheInsert {
    pub inserted_tokens: usize,
    pub replaced: Option<KvCacheHandle>,
    pub handle: KvCacheHandle,
}

#[derive(Debug, Default)]
pub struct RadixCache {
    nodes: Vec<Node>,
    capacity_tokens: Option<usize>,
    resident_tokens: usize,
    clock: u64,
}

#[derive(Debug, Default)]
struct Node {
    edge: Vec<TokenId>,
    children: BTreeMap<TokenId, NodeId>,
    handle: Option<KvCacheHandle>,
    last_access_tick: u64,
    hit_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NodeId(usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEviction {
    pub evicted: Vec<KvCacheHandle>,
}

impl RadixCache {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node::default()],
            capacity_tokens: None,
            resident_tokens: 0,
            clock: 1,
        }
    }

    pub fn with_capacity_tokens(capacity_tokens: usize) -> Self {
        let mut cache = Self::new();
        cache.capacity_tokens = Some(capacity_tokens);
        cache
    }

    pub fn insert(&mut self, tokens: &[TokenId], handle: KvCacheHandle) -> CacheInsert {
        assert_eq!(
            tokens.len(),
            handle.token_len,
            "KV handle length must match inserted prefix length"
        );

        self.clock = self.clock.saturating_add(1);
        self.nodes[0].last_access_tick = self.clock;
        self.nodes[0].hit_count = self.nodes[0].hit_count.saturating_add(1);
        let leaf = self.insert_from(NodeId(0), tokens);
        let replaced = self.nodes[leaf.0].handle.replace(handle);
        self.nodes[leaf.0].last_access_tick = self.clock;
        self.nodes[leaf.0].hit_count = self.nodes[leaf.0].hit_count.saturating_add(1);

        if let Some(old) = replaced {
            self.resident_tokens = self.resident_tokens.saturating_sub(old.token_len);
        }
        self.resident_tokens += handle.token_len;
        self.evict_to_capacity();

        CacheInsert {
            inserted_tokens: tokens.len(),
            replaced,
            handle,
        }
    }

    pub fn lookup(&mut self, tokens: &[TokenId]) -> CacheLookup {
        self.clock = self.clock.saturating_add(1);
        let access_tick = self.clock;
        let mut node_id = NodeId(0);
        let mut cursor = 0;
        let mut best = self.nodes[0].handle.map(|handle| (0, handle));
        self.nodes[0].last_access_tick = access_tick;
        self.nodes[0].hit_count = self.nodes[0].hit_count.saturating_add(1);

        while cursor < tokens.len() {
            let Some(child_id) = self.nodes[node_id.0].children.get(&tokens[cursor]).copied()
            else {
                break;
            };

            let child = &self.nodes[child_id.0];
            let matched = common_prefix_len(&tokens[cursor..], &child.edge);
            if matched < child.edge.len() {
                break;
            }

            cursor += matched;
            node_id = child_id;
            self.nodes[node_id.0].last_access_tick = access_tick;
            self.nodes[node_id.0].hit_count = self.nodes[node_id.0].hit_count.saturating_add(1);

            if let Some(handle) = self.nodes[node_id.0].handle {
                best = Some((cursor, handle));
            }
        }

        let (matched_tokens, handle) = best
            .map(|(matched_tokens, handle)| (matched_tokens, Some(handle)))
            .unwrap_or((0, None));

        CacheLookup {
            matched_tokens,
            handle,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.iter().filter(|node| node.handle.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn set_capacity_tokens(&mut self, capacity_tokens: Option<usize>) {
        self.capacity_tokens = capacity_tokens;
        self.evict_to_capacity();
    }

    pub fn capacity_tokens(&self) -> Option<usize> {
        self.capacity_tokens
    }

    pub fn resident_tokens(&self) -> usize {
        self.resident_tokens
    }

    pub fn evict_coldest(&mut self, tokens_to_evict: usize) -> CacheEviction {
        let mut handles: Vec<(u64, NodeId, KvCacheHandle)> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| node.handle.map(|handle| (node.last_access_tick, NodeId(idx), handle)))
            .collect();
        handles.sort_by_key(|(tick, _, _)| *tick);

        let mut evicted = Vec::new();
        let mut freed = 0usize;
        for (_, node_id, handle) in handles {
            if freed >= tokens_to_evict {
                break;
            }
            if self.nodes[node_id.0].handle == Some(handle) {
                self.nodes[node_id.0].handle = None;
                self.resident_tokens = self.resident_tokens.saturating_sub(handle.token_len);
                freed += handle.token_len;
                evicted.push(handle);
            }
        }

        CacheEviction { evicted }
    }

    fn insert_from(&mut self, parent_id: NodeId, tokens: &[TokenId]) -> NodeId {
        if tokens.is_empty() {
            return parent_id;
        }

        let first = tokens[0];
        let Some(child_id) = self.nodes[parent_id.0].children.get(&first).copied() else {
            let node_id = self.push_node(Node {
                edge: tokens.to_vec(),
                children: BTreeMap::new(),
                handle: None,
                last_access_tick: self.clock,
                hit_count: 0,
            });
            self.nodes[parent_id.0].children.insert(first, node_id);
            return node_id;
        };

        let common = common_prefix_len(tokens, &self.nodes[child_id.0].edge);
        let child_edge_len = self.nodes[child_id.0].edge.len();

        if common == child_edge_len {
            return self.insert_from(child_id, &tokens[common..]);
        }

        let split_id = self.split_child(parent_id, child_id, common);

        if common == tokens.len() {
            split_id
        } else {
            self.insert_from(split_id, &tokens[common..])
        }
    }

    fn split_child(&mut self, parent_id: NodeId, child_id: NodeId, split_at: usize) -> NodeId {
        debug_assert!(split_at > 0);
        debug_assert!(split_at < self.nodes[child_id.0].edge.len());

        let old_first = self.nodes[child_id.0].edge[0];
        let suffix = self.nodes[child_id.0].edge.split_off(split_at);
        let prefix = std::mem::replace(&mut self.nodes[child_id.0].edge, suffix);
        let new_first = self.nodes[child_id.0].edge[0];

        let split_id = self.push_node(Node {
            edge: prefix,
            children: BTreeMap::from([(new_first, child_id)]),
            handle: None,
            last_access_tick: self.clock,
            hit_count: self.nodes[child_id.0].hit_count,
        });

        self.nodes[parent_id.0].children.insert(old_first, split_id);
        split_id
    }

    fn push_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    fn evict_to_capacity(&mut self) {
        let Some(capacity) = self.capacity_tokens else {
            return;
        };

        if self.resident_tokens <= capacity {
            return;
        }

        let overflow = self.resident_tokens - capacity;
        let _ = self.evict_coldest(overflow);
    }
}

fn common_prefix_len(left: &[TokenId], right: &[TokenId]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle(block_id: u64, token_len: usize) -> KvCacheHandle {
        KvCacheHandle {
            block_id,
            token_len,
        }
    }

    #[test]
    fn returns_empty_lookup_on_miss() {
        let mut cache = RadixCache::new();

        assert_eq!(
            cache.lookup(&[1, 2, 3]),
            CacheLookup {
                matched_tokens: 0,
                handle: None
            }
        );
    }

    #[test]
    fn reuses_longest_cached_prefix() {
        let mut cache = RadixCache::new();
        cache.insert(&[10, 20], handle(1, 2));
        cache.insert(&[10, 20, 30, 40], handle(2, 4));

        assert_eq!(
            cache.lookup(&[10, 20, 30, 40, 50]),
            CacheLookup {
                matched_tokens: 4,
                handle: Some(handle(2, 4))
            }
        );
    }

    #[test]
    fn splits_edge_when_new_prefix_is_shorter_than_existing_path() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2, 3, 4], handle(7, 4));
        cache.insert(&[1, 2], handle(8, 2));

        assert_eq!(
            cache.lookup(&[1, 2, 9]),
            CacheLookup {
                matched_tokens: 2,
                handle: Some(handle(8, 2))
            }
        );
        assert_eq!(
            cache.lookup(&[1, 2, 3, 4, 5]),
            CacheLookup {
                matched_tokens: 4,
                handle: Some(handle(7, 4))
            }
        );
    }

    #[test]
    fn splits_edge_when_paths_diverge_mid_edge() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2, 3, 4], handle(11, 4));
        cache.insert(&[1, 2, 8, 9], handle(12, 4));

        assert_eq!(
            cache.lookup(&[1, 2, 3, 4]),
            CacheLookup {
                matched_tokens: 4,
                handle: Some(handle(11, 4))
            }
        );
        assert_eq!(
            cache.lookup(&[1, 2, 8, 9]),
            CacheLookup {
                matched_tokens: 4,
                handle: Some(handle(12, 4))
            }
        );
        assert_eq!(
            cache.lookup(&[1, 2, 0]),
            CacheLookup {
                matched_tokens: 0,
                handle: None
            }
        );
    }

    #[test]
    fn replaces_existing_handle_for_same_prefix() {
        let mut cache = RadixCache::new();
        cache.insert(&[4, 5, 6], handle(1, 3));

        let inserted = cache.insert(&[4, 5, 6], handle(2, 3));

        assert_eq!(inserted.replaced, Some(handle(1, 3)));
        assert_eq!(
            cache.lookup(&[4, 5, 6, 7]),
            CacheLookup {
                matched_tokens: 3,
                handle: Some(handle(2, 3))
            }
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn evicts_oldest_prefixes_when_capacity_is_exceeded() {
        let mut cache = RadixCache::with_capacity_tokens(5);
        cache.insert(&[1, 2], handle(1, 2));
        cache.insert(&[3, 4], handle(2, 2));
        cache.insert(&[5, 6], handle(3, 2));

        assert_eq!(cache.capacity_tokens(), Some(5));
        assert_eq!(cache.resident_tokens(), 4);
        assert!(cache.lookup(&[1, 2]).handle.is_none());
        assert_eq!(cache.lookup(&[3, 4]).handle, Some(handle(2, 2)));
        assert_eq!(cache.lookup(&[5, 6]).handle, Some(handle(3, 2)));
    }
}
