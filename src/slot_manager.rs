//! Slot-based request state management.
//!
//! Maps concurrent requests to fixed "slots" in GPU memory. Each slot
//! provides a stable index into a global page-mapping table, allowing
//! kernels to perform O(1) KV-cache lookups for any request in a batch.

use std::collections::VecDeque;

/// A unique identifier for a concurrent request slot.
///
/// These IDs are zero-indexed and correspond to the persistent page-table
/// entries in the GPU memory backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotId(pub u32);

/// Manages a pool of available request slots.
///
/// The `SlotPool` ensures that concurrent requests are isolated from each other
/// in GPU memory. It provides O(1) allocation and deallocation of slots.
#[derive(Debug)]
pub struct SlotPool {
    free_slots: VecDeque<u32>,
    max_slots: usize,
}

impl SlotPool {
    /// Creates a new pool with a fixed capacity.
    ///
    /// # Arguments
    /// * `max_slots` - The maximum number of concurrent requests the engine can handle.
    pub fn new(max_slots: usize) -> Self {
        let mut free_slots = VecDeque::with_capacity(max_slots);
        for i in 0..max_slots {
            free_slots.push_back(i as u32);
        }
        Self {
            free_slots,
            max_slots,
        }
    }

    /// Allocates a free slot from the pool. Returns `None` if all slots are occupied.
    ///
    /// The pool follows a FIFO strategy for slot reuse to minimize the chance of
    /// cache pollution across short-lived requests.
    pub fn alloc(&mut self) -> Option<SlotId> {
        self.free_slots.pop_front().map(SlotId)
    }

    /// Returns a slot to the pool for reuse.
    ///
    /// # Panics
    /// Panics if the `SlotId` is out of the valid range for this pool.
    pub fn release(&mut self, slot: SlotId) {
        assert!(
            (slot.0 as usize) < self.max_slots,
            "Invalid slot ID: {} is out of bounds for capacity {}",
            slot.0,
            self.max_slots
        );
        self.free_slots.push_back(slot.0);
    }

    /// Returns the number of currently available slots.
    pub fn available_slots(&self) -> usize {
        self.free_slots.len()
    }

    /// Returns the total capacity of the pool.
    pub fn capacity(&self) -> usize {
        self.max_slots
    }

    /// Returns true if the pool is empty (all slots allocated).
    pub fn is_empty(&self) -> bool {
        self.free_slots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_lifecycle() {
        let mut pool = SlotPool::new(4);
        assert_eq!(pool.available_slots(), 4);
        assert!(!pool.is_empty());

        let s0 = pool.alloc().unwrap();
        let s1 = pool.alloc().unwrap();
        assert_eq!(s0.0, 0);
        assert_eq!(s1.0, 1);
        assert_eq!(pool.available_slots(), 2);

        pool.release(s0);
        assert_eq!(pool.available_slots(), 3);

        let s2 = pool.alloc().unwrap();
        let s3 = pool.alloc().unwrap();
        let s4 = pool.alloc().unwrap();
        assert_eq!(s2.0, 2);
        assert_eq!(s3.0, 3);
        assert_eq!(s4.0, 0); // Reused s0

        assert!(pool.alloc().is_none());
        assert!(pool.is_empty());
    }

    #[test]
    #[should_panic(expected = "Invalid slot ID")]
    fn test_invalid_release() {
        let mut pool = SlotPool::new(2);
        pool.release(SlotId(5));
    }

    #[test]
    fn test_zero_capacity_pool() {
        let mut pool = SlotPool::new(0);
        assert_eq!(pool.capacity(), 0);
        assert_eq!(pool.available_slots(), 0);
        assert!(pool.is_empty());
        assert!(pool.alloc().is_none());
    }

    #[test]
    fn test_fifo_reuse() {
        let mut pool = SlotPool::new(3);
        let _ = pool.alloc(); // 0
        let s1 = pool.alloc().unwrap(); // 1
        let _ = pool.alloc(); // 2

        pool.release(s1);
        let s1_new = pool.alloc().unwrap();
        assert_eq!(s1_new.0, 1);
    }

    #[test]
    fn test_slot_id_traits() {
        let id = SlotId(1);
        let id2 = id.clone();
        assert_eq!(id, id2);
        assert!(format!("{:?}", id).contains("SlotId"));
    }
}
