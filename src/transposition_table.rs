use std::mem::size_of;

use crate::types::{
    EngineError,
    EngineResult,
    MATE_THRESHOLD,
    Move,
};

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
)]
#[allow(non_camel_case_types)]
pub enum BoundType {
    EXACT,
    LOWER,
    UPPER,
}

#[derive(
    Clone,
    Copy,
    Debug,
)]
pub struct TranspositionEntry {
    pub key: u64,
    pub best_move: Move,
    pub score: i16,
    pub depth_plies: i16,
    pub bound: BoundType,
    pub generation: u8,
}

impl Default for TranspositionEntry {
    fn default() -> Self {
        return Self {
            key: 0,
            best_move: Move::default(),
            score: 0,
            depth_plies: -1,
            bound: BoundType::UPPER,
            generation: 0,
        };
    }
}

pub struct TranspositionTable {
    entries: Vec<TranspositionEntry>,
    generation: u8,
}

impl TranspositionTable {
    pub fn create_table(size_mib: usize) -> EngineResult<Self> {
        let mut table = Self {
            entries: Vec::new(),
            generation: 0,
        };
        table.resize_table(size_mib)?;

        return Ok(table);
    }

    pub fn resize_table(
        &mut self,
        size_mib: usize,
    ) -> EngineResult<()> {
        if !(1..=512).contains(&size_mib) {
            return Err(EngineError::create_invalid_input("Failed to resize the hash table because its size must be between 1 and 512 MiB."));
        }

        let entry_count = (size_mib * 1024 * 1024) / size_of::<TranspositionEntry>();
        let mut replacement = Vec::new();
        replacement.try_reserve_exact(entry_count)
            .map_err(|_| EngineError::create_invalid_input("Failed to allocate the hash table."))?;
        replacement.resize(
            entry_count,
            TranspositionEntry::default(),
        );
        self.entries = replacement;
        self.generation = 0;

        return Ok(());
    }

    pub fn clear_table(&mut self) {
        self.entries.fill(TranspositionEntry::default());
        self.generation = 0;
    }

    pub fn start_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn probe_position(
        &self,
        key: u64,
    ) -> Option<TranspositionEntry> {
        let entry = self.entries[(key % self.entries.len() as u64) as usize];

        if (entry.depth_plies < 0) || (entry.key != key) {
            return None;
        }

        return Some(entry);
    }

    pub fn store_position(
        &mut self,
        key: u64,
        best_move: Move,
        mut score: i32,
        depth_plies: i32,
        bound: BoundType,
        ply: usize,
    ) {
        let index = (key % self.entries.len() as u64) as usize;
        let entry = &mut self.entries[index];

        if (entry.key != key) && (entry.generation == self.generation) && (entry.depth_plies as i32 > depth_plies) {
            return;
        }

        if score >= MATE_THRESHOLD {
            score += ply as i32;
        } else if score <= -MATE_THRESHOLD {
            score -= ply as i32;
        }

        *entry = TranspositionEntry {
            key,
            best_move,
            score: score as i16,
            depth_plies: depth_plies as i16,
            bound,
            generation: self.generation,
        };
    }

    pub fn decode_score(
        stored_score: i32,
        ply: usize,
    ) -> i32 {
        if stored_score >= MATE_THRESHOLD {
            return stored_score - ply as i32;
        }

        if stored_score <= -MATE_THRESHOLD {
            return stored_score + ply as i32;
        }

        return stored_score;
    }

    pub fn get_hashfull_per_mille(&self) -> i32 {
        let sample_count = self.entries.len().min(1000);
        let occupied_count = self.entries[..sample_count].iter()
            .filter(|entry| (entry.depth_plies >= 0) && (entry.generation == self.generation)).count();

        return ((occupied_count * 1000) / sample_count) as i32;
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{
        MATE_SCORE,
        Move,
    };
    use super::{
        BoundType,
        TranspositionTable,
    };

    #[test]
    fn test_collisions_keep_deeper_entries_until_generation_changes() {
        let mut table = TranspositionTable::create_table(1).unwrap();
        let first_key = 7;
        let colliding_key = first_key + table.entries.len() as u64;
        table.store_position(
            first_key,
            Move::default(),
            80,
            8,
            BoundType::EXACT,
            0,
        );
        table.store_position(
            colliding_key,
            Move::default(),
            -20,
            2,
            BoundType::UPPER,
            0,
        );
        assert!(table.probe_position(first_key).is_some());
        assert!(table.probe_position(colliding_key).is_none());
        table.start_generation();
        table.store_position(
            colliding_key,
            Move::default(),
            -20,
            2,
            BoundType::UPPER,
            0,
        );
        assert!(table.probe_position(first_key).is_none());
        assert_eq!(
            table.probe_position(colliding_key).unwrap().score,
            -20,
        );
    }

    #[test]
    fn test_resize_rejection_is_transactional_and_valid_resize_clears_entries() {
        let mut table = TranspositionTable::create_table(1).unwrap();
        table.store_position(
            0,
            Move::default(),
            42,
            3,
            BoundType::LOWER,
            0,
        );

        for size_mib in [0, 513, usize::MAX,] {
            assert!(table.resize_table(size_mib).is_err());
            assert_eq!(
                table.probe_position(0).unwrap().score,
                42,
            );
        }

        table.resize_table(2).unwrap();
        assert!(table.probe_position(0).is_none());
        assert_eq!(
            table.get_hashfull_per_mille(),
            0,
        );
    }

    #[test]
    fn test_hash_occupancy_clear_and_generation_wraparound() {
        let mut table = TranspositionTable::create_table(1).unwrap();

        for key in 0..1000 {
            table.store_position(
                key,
                Move::default(),
                0,
                1,
                BoundType::EXACT,
                0,
            );
        }

        assert_eq!(
            table.get_hashfull_per_mille(),
            1000,
        );
        table.start_generation();
        assert_eq!(
            table.get_hashfull_per_mille(),
            0,
        );

        for _ in 0..255 {
            table.start_generation();
        }

        assert!(table.probe_position(1).is_some());
        table.clear_table();
        assert!(table.probe_position(1).is_none());
        assert_eq!(
            table.get_hashfull_per_mille(),
            0,
        );
    }

    #[test]
    fn test_mate_scores_rebase_at_different_search_plies() {
        let mut table = TranspositionTable::create_table(1).unwrap();

        for (key, score, expected) in [(1, MATE_SCORE - 10, MATE_SCORE - 7,), (2, -MATE_SCORE + 10, -MATE_SCORE + 7,), (3, 320, 320,),] {
            table.store_position(
                key,
                Move::default(),
                score,
                4,
                BoundType::EXACT,
                5,
            );
            assert_eq!(
                TranspositionTable::decode_score(
                    table.probe_position(key).unwrap().score as i32,
                    2,
                ),
                expected,
            );
        }
    }
}
