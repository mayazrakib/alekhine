use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use crate::fingerprint::compute_byte_fingerprint;
use crate::position::{
    Position,
    UndoState,
};
use crate::tactical::can_attack_square;
use crate::types::{
    BOARD_SQUARE_COUNT,
    Color,
    EngineError,
    EngineResult,
    get_piece_index,
    Move,
};

pub const NEURAL_HIDDEN_COUNT: usize = 128;
pub const NEURAL_INPUT_COUNT: usize = NEURAL_HIDDEN_COUNT * 2;
pub const NEURAL_KING_BUCKET_COUNT: usize = 8;
pub const NEURAL_PIECE_FEATURE_COUNT: usize = 768 * NEURAL_KING_BUCKET_COUNT;
pub const NEURAL_FEATURE_COUNT: usize = NEURAL_PIECE_FEATURE_COUNT + 144;
pub const NEURAL_HEAD_COUNT: usize = 136;
pub const NEURAL_QUANTIZATION_SCALE: i32 = 256;
pub const NEURAL_SCORE_SCALE: i32 = 400;
pub const NEURAL_MAGIC: &[u8; 16] = b"ALEKHINE_NNUE_2\n";
pub const NEURAL_WEIGHT_LIMIT: i32 = 4096;
const MAXIMUM_FEATURE_COUNT: usize = BOARD_SQUARE_COUNT + 144;

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    PartialEq,
)]
pub struct SparseFeature {
    pub index: u16,
    pub count: u16,
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
)]
pub struct FeatureSet {
    pub features: [[SparseFeature; MAXIMUM_FEATURE_COUNT]; 2],
    pub counts: [usize; 2],
}

impl Default for FeatureSet {
    fn default() -> Self {
        return Self {
            features: [[SparseFeature::default(); MAXIMUM_FEATURE_COUNT]; 2],
            counts: [0; 2],
        };
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
)]
pub struct NeuralAccumulator {
    pub sums: [[i32; NEURAL_HIDDEN_COUNT]; 2],
    pub board: [i32; BOARD_SQUARE_COUNT],
    pub features: FeatureSet,
}

impl Default for NeuralAccumulator {
    fn default() -> Self {
        return Self {
            sums: [[0; NEURAL_HIDDEN_COUNT]; 2],
            board: [0; BOARD_SQUARE_COUNT],
            features: FeatureSet::default(),
        };
    }
}

#[derive(Clone)]
pub struct NeuralNetwork {
    pub feature_weights: Vec<i16>,
    pub hidden_biases: [i16; NEURAL_HIDDEN_COUNT],
    pub output_weights: Vec<i16>,
    pub output_biases: [i16; NEURAL_HEAD_COUNT],
    pub has_trained_value: bool,
    pub has_trained_policy: bool,
    pub has_trained_outcome: bool,
}

pub fn encode_features(board: &[i32; BOARD_SQUARE_COUNT]) -> FeatureSet {
    let mut feature_set = FeatureSet::default();
    let mut interactions = [[0u16; 12]; 12];

    for origin in 0..BOARD_SQUARE_COUNT {
        if board[origin] == 0 {
            continue;
        }

        for destination in 0..BOARD_SQUARE_COUNT {
            if (origin == destination) || (board[destination] == 0) {
                continue;
            }

            if can_attack_square(
                board,
                origin,
                destination,
            ) {
                interactions[get_piece_index(board[origin])][get_piece_index(board[destination])] += 1;
            }
        }
    }

    for perspective in 0..2 {
        let sign = if perspective == 0 {
            1
        } else {
            -1
        };
        let king_square = board.iter().position(|&piece| piece == sign * 6).unwrap_or(0);
        let oriented_king = king_square ^ (perspective * 56);
        let should_mirror = (oriented_king % 8) >= 4;
        let mirror_mask = usize::from(should_mirror) * 7;
        let canonical_king = oriented_king ^ mirror_mask;
        let king_bucket = ((canonical_king % 8) / 2) * 4 + (canonical_king / 16);
        let features = &mut feature_set.features[perspective];
        let mut count = 0;

        for (square, &piece) in board.iter().enumerate() {
            if piece == 0 {
                continue;
            }

            let oriented_square = square ^ (perspective * 56) ^ mirror_mask;
            features[count] = SparseFeature {
                index: ((king_bucket * 768) + (get_piece_index(piece * sign) * 64) + oriented_square) as u16,
                count: 1,
            };
            count += 1;
        }

        for (attacker, targets) in interactions.iter().enumerate() {
            for (target, &interaction_count) in targets.iter().enumerate() {
                if interaction_count == 0 {
                    continue;
                }

                let relative_attacker = (attacker + perspective * 6) % 12;
                let relative_target = (target + perspective * 6) % 12;
                features[count] = SparseFeature {
                    index: (NEURAL_PIECE_FEATURE_COUNT + relative_attacker * 12 + relative_target) as u16,
                    count: interaction_count,
                };
                count += 1;
            }
        }

        features[..count].sort_unstable_by_key(|feature| feature.index);
        feature_set.counts[perspective] = count;
    }

    return feature_set;
}

pub fn get_policy_head_indices(
    chess_move: Move,
    board: &[i32; BOARD_SQUARE_COUNT],
    side: Color,
) -> [usize; 3] {
    let perspective = usize::from(side == Color::BLACK);
    let sign = if perspective == 0 {
        1
    } else {
        -1
    };
    let king_square = board.iter().position(|&piece| piece == sign * 6).unwrap_or(0);
    let orientation_mask = perspective * 56;
    let mirror_mask = usize::from(((king_square ^ orientation_mask) % 8) >= 4) * 7;
    let square_mask = orientation_mask ^ mirror_mask;
    let promotion = chess_move.get_promotion() as usize;
    let promotion_head = if promotion >= 2 {
        132 + promotion - 2
    } else {
        0
    };

    return [4 + (chess_move.get_origin() as usize ^ square_mask), 68 + (chess_move.get_destination() as usize ^ square_mask), promotion_head,];
}

fn read_integer(
    payload: &[u8],
    offset: &mut usize,
) -> EngineResult<u32> {
    let bytes: [u8; 4] = payload.get(*offset..(*offset + 4))
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| EngineError::create_invalid_input("Failed to load the neural network because its header is truncated."))?;
    *offset += 4;

    return Ok(u32::from_le_bytes(bytes));
}

impl NeuralNetwork {
    pub fn create_zero_network() -> Self {
        return Self {
            feature_weights: vec![0; NEURAL_FEATURE_COUNT * NEURAL_HIDDEN_COUNT],
            hidden_biases: [0; NEURAL_HIDDEN_COUNT],
            output_weights: vec![0; NEURAL_HEAD_COUNT * NEURAL_INPUT_COUNT],
            output_biases: [0; NEURAL_HEAD_COUNT],
            has_trained_value: false,
            has_trained_policy: false,
            has_trained_outcome: false,
        };
    }

    fn apply_feature(
        &self,
        sums: &mut [i32; NEURAL_HIDDEN_COUNT],
        feature_index: usize,
        count: i32,
    ) {
        let offset = feature_index * NEURAL_HIDDEN_COUNT;

        for (channel, activation) in sums.iter_mut().enumerate() {
            *activation += self.feature_weights[offset + channel] as i32 * count;
        }
    }

    pub fn build_accumulator(
        &self,
        position: &Position,
    ) -> NeuralAccumulator {
        let mut accumulator = NeuralAccumulator::default();

        for (square, piece) in accumulator.board.iter_mut().enumerate() {
            *piece = position.get_piece(square as i32);
        }

        accumulator.features = encode_features(&accumulator.board);

        for perspective in 0..2 {
            accumulator.sums[perspective] = self.hidden_biases.map(i32::from);

            for feature in &accumulator.features.features[perspective][..accumulator.features.counts[perspective]] {
                self.apply_feature(
                    &mut accumulator.sums[perspective],
                    feature.index as usize,
                    feature.count as i32,
                );
            }
        }

        return accumulator;
    }

    pub fn update_accumulator(
        &self,
        previous: &NeuralAccumulator,
        undo: &UndoState,
    ) -> NeuralAccumulator {
        let mut accumulator = *previous;

        for change in &undo.changes[..undo.change_count] {
            accumulator.board[change.square as usize] = change.after;
        }

        accumulator.features = encode_features(&accumulator.board);

        for perspective in 0..2 {
            let previous_features = &previous.features.features[perspective][..previous.features.counts[perspective]];
            let next_features = &accumulator.features.features[perspective][..accumulator.features.counts[perspective]];
            let mut previous_index = 0;
            let mut next_index = 0;

            while (previous_index < previous_features.len()) || (next_index < next_features.len()) {
                let previous_feature = previous_features.get(previous_index).copied().unwrap_or(SparseFeature {
                    index: u16::MAX,
                    count: 0,
                });
                let next_feature = next_features.get(next_index).copied().unwrap_or(SparseFeature {
                    index: u16::MAX,
                    count: 0,
                });
                let feature_index = previous_feature.index.min(next_feature.index);
                let mut delta = 0;

                if previous_feature.index == feature_index {
                    delta -= previous_feature.count as i32;
                    previous_index += 1;
                }

                if next_feature.index == feature_index {
                    delta += next_feature.count as i32;
                    next_index += 1;
                }

                if delta != 0 {
                    self.apply_feature(
                        &mut accumulator.sums[perspective],
                        feature_index as usize,
                        delta,
                    );
                }
            }
        }

        return accumulator;
    }

    pub fn evaluate_head(
        &self,
        accumulator: &NeuralAccumulator,
        side: Color,
        head: usize,
    ) -> f64 {
        let first_perspective = usize::from(side == Color::BLACK);
        let mut total = self.output_biases[head] as i64
            * NEURAL_QUANTIZATION_SCALE as i64 * NEURAL_QUANTIZATION_SCALE as i64;

        for input in 0..NEURAL_INPUT_COUNT {
            let perspective = first_perspective ^ (input / NEURAL_HIDDEN_COUNT);
            let activation = accumulator.sums[perspective][input % NEURAL_HIDDEN_COUNT].clamp(
                0,
                NEURAL_QUANTIZATION_SCALE,
            ) as i64;
            total += activation * activation * self.output_weights[head * NEURAL_INPUT_COUNT + input] as i64;
        }

        return total as f64 / (NEURAL_QUANTIZATION_SCALE as f64).powi(3);
    }

    pub fn evaluate_policy(
        &self,
        accumulator: &NeuralAccumulator,
        side: Color,
        chess_move: Move,
    ) -> f64 {
        let indices = get_policy_head_indices(
            chess_move,
            &accumulator.board,
            side,
        );
        let mut logit = self.evaluate_head(
            accumulator,
            side,
            indices[0],
        ) + self.evaluate_head(
            accumulator,
            side,
            indices[1],
        );

        if indices[2] != 0 {
            logit += self.evaluate_head(
                accumulator,
                side,
                indices[2],
            );
        }

        return logit;
    }

    pub fn get_fingerprint(&self) -> u64 {
        return compute_byte_fingerprint(&self.serialize_network());
    }

    fn serialize_network(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(NEURAL_MAGIC);

        for integer in [NEURAL_FEATURE_COUNT, NEURAL_HIDDEN_COUNT, NEURAL_HEAD_COUNT, NEURAL_QUANTIZATION_SCALE as usize, NEURAL_SCORE_SCALE as usize,] {
            payload.extend_from_slice(&(integer as u32).to_le_bytes());
        }

        let flags = u32::from(self.has_trained_value)
            | (u32::from(self.has_trained_policy) << 1)
            | (u32::from(self.has_trained_outcome) << 2);
        payload.extend_from_slice(&flags.to_le_bytes());

        for weights in [&self.feature_weights[..], &self.hidden_biases[..], &self.output_weights[..], &self.output_biases[..],] {
            for weight in weights {
                payload.extend_from_slice(&weight.to_le_bytes());
            }
        }

        return payload;
    }

    pub fn save_network(
        &self,
        path: &Path,
    ) -> EngineResult<()> {
        let mut payload = self.serialize_network();
        let fingerprint = compute_byte_fingerprint(&payload);
        payload.extend_from_slice(&fingerprint.to_le_bytes());
        let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
        output.write_all(&payload)?;
        output.flush()?;

        return Ok(());
    }

    pub fn load_network(payload: &[u8]) -> EngineResult<Self> {
        let weight_count = (NEURAL_FEATURE_COUNT * NEURAL_HIDDEN_COUNT) + NEURAL_HIDDEN_COUNT
            + (NEURAL_HEAD_COUNT * NEURAL_INPUT_COUNT) + NEURAL_HEAD_COUNT;
        let expected_size = NEURAL_MAGIC.len() + 24 + (weight_count * 2) + 8;

        if (payload.len() != expected_size) || !payload.starts_with(NEURAL_MAGIC) {
            return Err(EngineError::create_invalid_input("Failed to load the neural network because its size or identifier is invalid."));
        }

        let checksum_offset = payload.len() - 8;
        let checksum_bytes: [u8; 8] = payload[checksum_offset..].try_into()
            .map_err(|_| EngineError::create_invalid_input("Failed to load the neural checksum."))?;

        if compute_byte_fingerprint(&payload[..checksum_offset]) != u64::from_le_bytes(checksum_bytes) {
            return Err(EngineError::create_invalid_input("Failed to load the neural network because its checksum is invalid."));
        }

        let mut offset = NEURAL_MAGIC.len();

        for expected in [NEURAL_FEATURE_COUNT, NEURAL_HIDDEN_COUNT, NEURAL_HEAD_COUNT, NEURAL_QUANTIZATION_SCALE as usize, NEURAL_SCORE_SCALE as usize,] {
            if read_integer(
                payload,
                &mut offset,
            )? as usize != expected {
                return Err(EngineError::create_invalid_input("Failed to load the neural network because its architecture is incompatible."));
            }
        }

        let flags = read_integer(
            payload,
            &mut offset,
        )?;

        if flags > 7 {
            return Err(EngineError::create_invalid_input("Failed to load the neural network because its training flags are invalid."));
        }

        let mut network = Self::create_zero_network();
        network.has_trained_value = (flags & 1) != 0;
        network.has_trained_policy = (flags & 2) != 0;
        network.has_trained_outcome = (flags & 4) != 0;

        for weights in [&mut network.feature_weights[..], &mut network.hidden_biases[..], &mut network.output_weights[..], &mut network.output_biases[..],] {
            for weight in weights {
                let bytes: [u8; 2] = payload[offset..(offset + 2)].try_into()
                    .map_err(|_| EngineError::create_invalid_input("Failed to load a neural weight."))?;
                *weight = i16::from_le_bytes(bytes);
                offset += 2;

                if (*weight as i32).abs() > NEURAL_WEIGHT_LIMIT {
                    return Err(EngineError::create_invalid_input("Failed to load the neural network because a weight exceeds its arithmetic bound."));
                }
            }
        }

        return Ok(network);
    }
}

#[cfg(test)]
mod tests {
    use crate::fingerprint::compute_byte_fingerprint;
    use crate::position::Position;
    use crate::types::{
        Color,
        START_FEN,
    };
    use super::{
        encode_features,
        get_policy_head_indices,
        NEURAL_HEAD_COUNT,
        NEURAL_HIDDEN_COUNT,
        NEURAL_INPUT_COUNT,
        NEURAL_MAGIC,
        NEURAL_QUANTIZATION_SCALE,
        NEURAL_WEIGHT_LIMIT,
        NeuralAccumulator,
        NeuralNetwork,
    };

    fn create_payload(network: &NeuralNetwork) -> Vec<u8> {
        let mut payload = network.serialize_network();
        let fingerprint = compute_byte_fingerprint(&payload);
        payload.extend_from_slice(&fingerprint.to_le_bytes());

        return payload;
    }

    #[test]
    fn test_binary_architecture_rejects_rechecksummed_incompatible_headers() {
        let network = NeuralNetwork::create_zero_network();

        for offset in (NEURAL_MAGIC.len()..NEURAL_MAGIC.len() + 20).step_by(4) {
            let mut payload = network.serialize_network();
            payload[offset] ^= 1;
            let fingerprint = compute_byte_fingerprint(&payload);
            payload.extend_from_slice(&fingerprint.to_le_bytes());
            let error = NeuralNetwork::load_network(&payload).err().unwrap();
            assert!(error.to_string().contains("architecture"));
        }
    }

    #[test]
    fn test_binary_flags_roundtrip_and_reject_unknown_bits() {
        for flags in 0..8 {
            let mut network = NeuralNetwork::create_zero_network();
            network.has_trained_value = (flags & 1) != 0;
            network.has_trained_policy = (flags & 2) != 0;
            network.has_trained_outcome = (flags & 4) != 0;
            let loaded = NeuralNetwork::load_network(&create_payload(&network)).unwrap();
            assert_eq!(
                (loaded.has_trained_value, loaded.has_trained_policy, loaded.has_trained_outcome,),
                (network.has_trained_value, network.has_trained_policy, network.has_trained_outcome,),
            );
        }

        let mut payload = NeuralNetwork::create_zero_network().serialize_network();
        payload[NEURAL_MAGIC.len() + 20] = 8;
        let fingerprint = compute_byte_fingerprint(&payload);
        payload.extend_from_slice(&fingerprint.to_le_bytes());
        assert!(NeuralNetwork::load_network(&payload).err().unwrap().to_string().contains("flags"));
    }

    #[test]
    fn test_binary_rejects_oversized_weights_even_with_valid_checksums() {
        for weight in [i16::MIN, -(NEURAL_WEIGHT_LIMIT as i16) - 1, NEURAL_WEIGHT_LIMIT as i16 + 1, i16::MAX,] {
            let mut network = NeuralNetwork::create_zero_network();
            network.feature_weights[0] = weight;
            assert!(NeuralNetwork::load_network(&create_payload(&network)).err().unwrap().to_string().contains("weight"));
        }
    }

    #[test]
    fn test_binary_rejects_truncated_appended_and_wrong_magic_payloads() {
        let payload = create_payload(&NeuralNetwork::create_zero_network());

        for length in [0, 15, 16, 39, 40, payload.len() - 1,] {
            assert!(NeuralNetwork::load_network(&payload[..length]).is_err());
        }

        let mut appended = payload.clone();
        appended.push(0);
        assert!(NeuralNetwork::load_network(&appended).is_err());
        let mut wrong_magic = payload;
        wrong_magic[0] ^= 1;
        assert!(NeuralNetwork::load_network(&wrong_magic).is_err());
    }

    #[test]
    fn test_integer_inference_clips_squares_and_orders_perspectives() {
        let mut network = NeuralNetwork::create_zero_network();
        let mut accumulator = NeuralAccumulator::default();
        accumulator.sums[0][0] = -256;
        accumulator.sums[0][1] = NEURAL_QUANTIZATION_SCALE / 2;
        accumulator.sums[0][2] = NEURAL_QUANTIZATION_SCALE * 2;
        accumulator.sums[1][0] = NEURAL_QUANTIZATION_SCALE;
        network.output_biases[0] = NEURAL_QUANTIZATION_SCALE as i16;
        network.output_weights[0] = 1024;
        network.output_weights[1] = 256;
        network.output_weights[2] = 512;
        network.output_weights[NEURAL_HIDDEN_COUNT] = 768;
        assert_eq!(
            network.evaluate_head(
                &accumulator,
                Color::WHITE,
                0,
            ),
            6.25,
        );
        assert_eq!(
            network.evaluate_head(
                &accumulator,
                Color::BLACK,
                0,
            ),
            5.0,
        );
    }

    #[test]
    fn test_every_head_uses_its_own_weights_and_bias() {
        let mut network = NeuralNetwork::create_zero_network();
        let mut accumulator = NeuralAccumulator::default();
        accumulator.sums[0][0] = NEURAL_QUANTIZATION_SCALE;

        for head in 0..NEURAL_HEAD_COUNT {
            network.output_biases[head] = head as i16;
            network.output_weights[head * NEURAL_INPUT_COUNT] = head as i16;
            assert_eq!(
                network.evaluate_head(
                    &accumulator,
                    Color::WHITE,
                    head,
                ),
                head as f64 / 128.0,
            );
        }
    }

    #[test]
    fn test_promotion_policy_heads_are_distinct_and_in_bounds() {
        let mut position = Position::parse_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let accumulator = NeuralNetwork::create_zero_network().build_accumulator(&position);

        for (move_name, expected_head) in [("a7a8n", 132,), ("a7a8b", 133,), ("a7a8r", 134,), ("a7a8q", 135,),] {
            let indices = get_policy_head_indices(
                position.find_legal_move(move_name).unwrap(),
                &accumulator.board,
                Color::WHITE,
            );
            assert_eq!(
                indices[2],
                expected_head,
            );
            assert!(indices.iter().all(|&index| index < NEURAL_HEAD_COUNT));
        }
    }

    #[test]
    fn test_sparse_features_are_sorted_unique_and_preserve_interaction_counts() {
        let position = Position::parse_fen(START_FEN).unwrap();
        let accumulator = NeuralNetwork::create_zero_network().build_accumulator(&position);
        let features = encode_features(&accumulator.board);

        for perspective in 0..2 {
            let active = &features.features[perspective][..features.counts[perspective]];
            assert!(active.windows(2).all(|pair| pair[0].index < pair[1].index));
            assert!(active.iter().all(|feature| feature.count > 0));
            assert!(active.iter().any(|feature| feature.count > 1));
        }
    }
}
