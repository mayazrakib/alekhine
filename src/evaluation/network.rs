use std::fs::File;
use std::io::{
    BufWriter,
    Read,
    Write,
};
use std::path::Path;
use std::sync::Arc;

use crate::chess::fingerprint::{
    FINGERPRINT_OFFSET,
    update_byte_fingerprint,
};
use crate::evaluation::neural::{
    get_policy_head_indices,
    NEURAL_HEAD_COUNT,
    NEURAL_MAGIC,
    NEURAL_SCORE_SCALE,
    NeuralAccumulator,
    NeuralNetwork,
};
use crate::chess::position::{
    Position,
    UndoState,
};
use crate::evaluation::stockfish_network::{
    STOCKFISH_VERSION,
    StockfishNetwork,
};
use crate::chess::types::{
    BOARD_SQUARE_COUNT,
    Color,
    EngineError,
    EngineResult,
    get_piece_index,
    get_piece_score,
    get_piece_type,
    MAXIMUM_MOVE_COUNT,
    Move,
    MoveFlag,
    MoveList,
    PIECE_SCORES,
    PieceType,
};

pub const NETWORK_FEATURE_COUNT: usize = 768;
pub const NETWORK_HIDDEN_COUNT: usize = 32;
pub const NETWORK_QUANTIZATION_SCALE: i32 = 256;
const NETWORK_ACTIVATION_LIMIT: i32 = 32767;
const NETWORK_WEIGHT_LIMIT: i32 = 4096;
const NETWORK_BIAS_LIMIT: i32 = 1000000;
const MAXIMUM_NETWORK_BYTES: u64 = 2 * 1024 * 1024;

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
)]
pub struct Accumulator {
    pub bootstrap: [i32; NETWORK_HIDDEN_COUNT],
    pub neural: NeuralAccumulator,
}

type HiddenWeights = [i16; NETWORK_HIDDEN_COUNT];
type FeatureWeights = [HiddenWeights; NETWORK_FEATURE_COUNT];
type PolicyWeights = [HiddenWeights; BOARD_SQUARE_COUNT];

#[derive(
    Clone,
    Copy,
    Debug,
)]
pub struct OutcomeProbabilities {
    pub win: f64,
    pub draw: f64,
    pub loss: f64,
}

#[derive(Clone)]
pub struct Network {
    feature_weights: FeatureWeights,
    hidden_biases: [i32; NETWORK_HIDDEN_COUNT],
    value_weights: HiddenWeights,
    origin_weights: PolicyWeights,
    destination_weights: PolicyWeights,
    origin_biases: [i16; BOARD_SQUARE_COUNT],
    destination_biases: [i16; BOARD_SQUARE_COUNT],
    value_bias: i32,
    has_trained_weights: bool,
    neural_network: Option<NeuralNetwork>,
    stockfish_network: Option<Arc<StockfishNetwork>>,
}

fn get_bootstrap_feature_score(
    piece: i32,
    square: i32,
) -> i32 {
    let piece_type = get_piece_type(piece);

    let rank = if piece > 0 {
        square / 8
    } else {
        7 - (square / 8)
    };

    let file = square % 8;
    let center_score = 7 - ((file * 2) - 7).abs() - ((rank * 2) - 7).abs();

    let positional_score = match piece_type {
        PieceType::PAWN => (rank * 7) + (center_score * 2),
        PieceType::KNIGHT | PieceType::BISHOP => 35 + (center_score * 6),
        PieceType::ROOK => rank * 2,
        PieceType::KING => 40 - (rank * 5) - (center_score * 2),
        _ => 0,
    };

    return get_piece_score(piece) + positional_score;
}

fn read_weight(
    tokens: &mut std::str::SplitWhitespace<'_>,
    limit: i32,
) -> EngineResult<i32> {
    let weight = tokens.next().and_then(|token| token.parse::<i32>().ok())
        .ok_or_else(|| EngineError::create_invalid_input("Failed to load the network because a weight is invalid."))?;

    if (weight < -limit) || (weight > limit) {
        return Err(EngineError::create_invalid_input("Failed to load the network because a weight is invalid."));
    }

    return Ok(weight);
}

fn update_fingerprint(
    fingerprint: &mut u64,
    weight: i32,
) {
    update_byte_fingerprint(
        fingerprint,
        &weight.to_le_bytes(),
    );
}

impl Default for Network {
    fn default() -> Self {
        return Self::create_network();
    }
}

impl Network {
    pub fn create_network() -> Self {
        let mut network = Self {
            feature_weights: [[0; NETWORK_HIDDEN_COUNT]; NETWORK_FEATURE_COUNT],
            hidden_biases: [0; NETWORK_HIDDEN_COUNT],
            value_weights: [0; NETWORK_HIDDEN_COUNT],
            origin_weights: [[0; NETWORK_HIDDEN_COUNT]; BOARD_SQUARE_COUNT],
            destination_weights: [[0; NETWORK_HIDDEN_COUNT]; BOARD_SQUARE_COUNT],
            origin_biases: [0; BOARD_SQUARE_COUNT],
            destination_biases: [0; BOARD_SQUARE_COUNT],
            value_bias: 0,
            has_trained_weights: false,
            neural_network: None,
            stockfish_network: None,
        };

        for piece_index in 0..12 {
            let piece = if piece_index < 6 {
                piece_index as i32 + 1
            } else {
                -(piece_index as i32 - 5)
            };

            let hidden_index = usize::from(piece < 0);

            for square in 0..BOARD_SQUARE_COUNT {
                network.feature_weights[(piece_index * 64) + square][hidden_index] = get_bootstrap_feature_score(
                    piece,
                    square as i32,
                ) as i16;
            }
        }

        network.value_weights[0] = NETWORK_QUANTIZATION_SCALE as i16;
        network.value_weights[1] = -NETWORK_QUANTIZATION_SCALE as i16;

        for square in 0..BOARD_SQUARE_COUNT {
            let file = square as i32 % 8;
            let rank = square as i32 / 8;
            network.destination_biases[square] = ((7 - ((file * 2) - 7).abs() - ((rank * 2) - 7).abs()) * 16) as i16;
        }

        return network;
    }

    fn apply_feature(
        &self,
        accumulator: &mut [i32; NETWORK_HIDDEN_COUNT],
        piece: i32,
        square: i32,
        direction: i32,
    ) {
        if piece == 0 {
            return;
        }

        let weights = &self.feature_weights[(get_piece_index(piece) * 64) + square as usize];

        for index in 0..NETWORK_HIDDEN_COUNT {
            accumulator[index] += direction * weights[index] as i32;
        }
    }

    pub fn build_accumulator(
        &self,
        position: &Position,
    ) -> Accumulator {
        let mut accumulator = Accumulator {
            bootstrap: self.hidden_biases,
            neural: NeuralAccumulator::default(),
        };

        for square in 0..BOARD_SQUARE_COUNT as i32 {
            self.apply_feature(
                &mut accumulator.bootstrap,
                position.get_piece(square),
                square,
                1,
            );
        }

        if let Some(network) = &self.neural_network {
            accumulator.neural = network.build_accumulator(position);
        }

        return accumulator;
    }

    pub fn update_accumulator(
        &self,
        previous: &Accumulator,
        undo: &UndoState,
    ) -> Accumulator {
        let mut accumulator = *previous;

        for change in &undo.changes[..undo.change_count] {
            self.apply_feature(
                &mut accumulator.bootstrap,
                change.before,
                change.square,
                -1,
            );
            self.apply_feature(
                &mut accumulator.bootstrap,
                change.after,
                change.square,
                1,
            );
        }

        if let Some(network) = &self.neural_network {
            accumulator.neural = network.update_accumulator(
                &previous.neural,
                undo,
            );
        }

        return accumulator;
    }

    pub fn evaluate_position(
        &self,
        position: &Position,
        accumulator: &Accumulator,
    ) -> i32 {
        if let Some(network) = &self.stockfish_network {
            return network.evaluate_position(position);
        }

        if let Some(network) = &self.neural_network {
            if network.has_trained_value {
                return (network.evaluate_head(
                    &accumulator.neural,
                    position.get_side_to_move(),
                    0,
                ) * NEURAL_SCORE_SCALE as f64).round().clamp(
                    -20000.0,
                    20000.0,
                ) as i32;
            }
        }

        let mut score = self.value_bias as i64;

        for (index, &hidden_activation) in accumulator.bootstrap.iter().enumerate() {
            let activation = hidden_activation.clamp(
                0,
                NETWORK_ACTIVATION_LIMIT,
            );
            score += activation as i64 * self.value_weights[index] as i64;
        }

        score /= NETWORK_QUANTIZATION_SCALE as i64;
        score = score.clamp(
            -20000,
            20000,
        );

        let side_score = if position.get_side_to_move() == Color::WHITE {
            score
        } else {
            -score
        };

        return side_score as i32 + 10;
    }

    pub fn evaluate_policy(
        &self,
        position: &Position,
        accumulator: &Accumulator,
        chess_move: Move,
    ) -> f64 {
        if let Some(network) = &self.neural_network {
            if network.has_trained_policy {
                return network.evaluate_policy(
                    &accumulator.neural,
                    position.get_side_to_move(),
                    chess_move,
                ).clamp(
                    -30.0,
                    30.0,
                );
            }
        }

        let origin = chess_move.get_origin() as usize;
        let destination = chess_move.get_destination() as usize;
        let mut policy_score = self.origin_biases[origin] as i64 + self.destination_biases[destination] as i64;

        for (index, &hidden_activation) in accumulator.bootstrap.iter().enumerate() {
            let activation = hidden_activation.clamp(
                0,
                NETWORK_ACTIVATION_LIMIT,
            );
            let weight = self.origin_weights[origin][index] as i32 + self.destination_weights[destination][index] as i32;
            policy_score += (activation as i64 * weight as i64) / NETWORK_QUANTIZATION_SCALE as i64;
        }

        let mut logit = policy_score as f64 / NETWORK_QUANTIZATION_SCALE as f64;

        if !self.has_trained_weights || self.neural_network.is_some() {
            let attacker_score = get_piece_score(position.get_piece(chess_move.get_origin()));

            let victim_score = if chess_move.has_flag(MoveFlag::EN_PASSANT) {
                PIECE_SCORES[1]
            } else {
                get_piece_score(position.get_piece(chess_move.get_destination()))
            };

            logit += (victim_score as f64 / 180.0) - if chess_move.has_flag(MoveFlag::CAPTURE) {
                attacker_score as f64 / 1200.0
            } else {
                0.0
            };
            logit += PIECE_SCORES[chess_move.get_promotion() as usize] as f64 / 250.0;
        }

        return logit.clamp(
            -30.0,
            30.0,
        );
    }

    pub fn is_trained(&self) -> bool {
        if self.stockfish_network.is_some() {
            return true;
        }

        if let Some(network) = &self.neural_network {
            return network.has_trained_value;
        }

        return self.has_trained_weights;
    }

    pub fn evaluate_policies(
        &self,
        position: &Position,
        accumulator: &Accumulator,
        moves: &MoveList,
    ) -> [f64; MAXIMUM_MOVE_COUNT] {
        let mut scores = [0.0; MAXIMUM_MOVE_COUNT];

        if let Some(network) = &self.neural_network {
            if network.has_trained_policy {
                let mut heads = [0.0; NEURAL_HEAD_COUNT];

                for (head, logit) in heads.iter_mut().enumerate().skip(4) {
                    *logit = network.evaluate_head(
                        &accumulator.neural,
                        position.get_side_to_move(),
                        head,
                    );
                }

                for (index, &chess_move) in moves.moves[..moves.count].iter().enumerate() {
                    let indices = get_policy_head_indices(
                        chess_move,
                        &accumulator.neural.board,
                        position.get_side_to_move(),
                    );
                    scores[index] = heads[indices[0]] + heads[indices[1]];

                    if indices[2] != 0 {
                        scores[index] += heads[indices[2]];
                    }

                    scores[index] = scores[index].clamp(
                        -30.0,
                        30.0,
                    );
                }

                return scores;
            }
        }

        for (index, &chess_move) in moves.moves[..moves.count].iter().enumerate() {
            scores[index] = self.evaluate_policy(
                position,
                accumulator,
                chess_move,
            );
        }

        return scores;
    }

    pub fn create_neural_network(network: NeuralNetwork) -> Self {
        let mut evaluator = Self::create_network();
        evaluator.neural_network = Some(network);

        return evaluator;
    }

    pub fn create_stockfish_network(network: StockfishNetwork) -> Self {
        let mut evaluator = Self::create_network();
        evaluator.stockfish_network = Some(Arc::new(network));

        return evaluator;
    }

    pub fn evaluate_expected_outcome(
        &self,
        position: &Position,
        accumulator: &Accumulator,
    ) -> f64 {
        if let Some(network) = &self.neural_network {
            if network.has_trained_outcome {
                let mut logits = [0.0; 3];

                for (index, logit) in logits.iter_mut().enumerate() {
                    *logit = network.evaluate_head(
                        &accumulator.neural,
                        position.get_side_to_move(),
                        index + 1,
                    );
                }

                let maximum = logits.iter().copied().fold(
                    f64::NEG_INFINITY,
                    f64::max,
                );
                let probabilities = logits.map(|logit| (logit - maximum).exp());
                let total: f64 = probabilities.iter().sum();

                return (probabilities[0] - probabilities[2]) / total;
            }
        }

        return (self.evaluate_position(
            position,
            accumulator,
        ) as f64 / NEURAL_SCORE_SCALE as f64).tanh();
    }

    pub fn get_fingerprint(&self) -> u64 {
        if let Some(network) = &self.stockfish_network {
            return network.get_fingerprint();
        }

        if let Some(network) = &self.neural_network {
            return network.get_fingerprint();
        }

        let mut fingerprint = FINGERPRINT_OFFSET;
        update_fingerprint(
            &mut fingerprint,
            i32::from(self.has_trained_weights),
        );

        for &weight in self.feature_weights.iter().flatten() {
            update_fingerprint(
                &mut fingerprint,
                weight as i32,
            );
        }

        for &bias in &self.hidden_biases {
            update_fingerprint(
                &mut fingerprint,
                bias,
            );
        }

        for &weight in &self.value_weights {
            update_fingerprint(
                &mut fingerprint,
                weight as i32,
            );
        }

        for head in [&self.origin_weights, &self.destination_weights,] {
            for &weight in head.iter().flatten() {
                update_fingerprint(
                    &mut fingerprint,
                    weight as i32,
                );
            }
        }

        for biases in [&self.origin_biases, &self.destination_biases,] {
            for &bias in biases {
                update_fingerprint(
                    &mut fingerprint,
                    bias as i32,
                );
            }
        }

        update_fingerprint(
            &mut fingerprint,
            self.value_bias,
        );

        return fingerprint;
    }

    pub fn load_network(path: &Path) -> EngineResult<Self> {
        let mut input = File::open(path)?;
        let mut prefix = Vec::new();
        Read::by_ref(&mut input).take(4).read_to_end(&mut prefix)?;

        if prefix == STOCKFISH_VERSION.to_le_bytes() {
            return Ok(Self::create_stockfish_network(StockfishNetwork::load_network(path)?));
        }

        if input.metadata()?.len() > MAXIMUM_NETWORK_BYTES {
            return Err(EngineError::create_invalid_input("Failed to load the network because its size exceeds two MiB."));
        }

        let mut bytes = prefix;
        input.take(MAXIMUM_NETWORK_BYTES + 1).read_to_end(&mut bytes)?;

        if bytes.len() as u64 > MAXIMUM_NETWORK_BYTES {
            return Err(EngineError::create_invalid_input("Failed to load the network because its size exceeds two MiB."));
        }

        if bytes.starts_with(NEURAL_MAGIC) {
            return Ok(Self::create_neural_network(NeuralNetwork::load_network(&bytes)?));
        }

        let payload = String::from_utf8(bytes).map_err(|_| EngineError::create_invalid_input("Failed to decode the network payload."))?;
        let mut tokens = payload.split_whitespace();
        let magic = tokens.next();
        let version = tokens.next().and_then(|token| token.parse::<u32>().ok());
        let feature_count = tokens.next().and_then(|token| token.parse::<usize>().ok());
        let hidden_count = tokens.next().and_then(|token| token.parse::<usize>().ok());
        let trained_flag = tokens.next().and_then(|token| token.parse::<u8>().ok());
        let expected_fingerprint = tokens.next().and_then(|token| token.parse::<u64>().ok());

        let has_valid_header = (magic == Some("ALEKHINE_NETWORK"))
            && (version == Some(1))
            && (feature_count == Some(NETWORK_FEATURE_COUNT))
            && (hidden_count == Some(NETWORK_HIDDEN_COUNT))
            && matches!(
                trained_flag,
                Some(0 | 1),
            )
            && expected_fingerprint.is_some();

        if !has_valid_header {
            return Err(EngineError::create_invalid_input("Failed to load the network because its header is incompatible."));
        }

        let mut network = Self::create_network();
        network.has_trained_weights = trained_flag == Some(1);

        for weight in network.feature_weights.iter_mut().flatten() {
            *weight = read_weight(
                &mut tokens,
                NETWORK_WEIGHT_LIMIT,
            )? as i16;
        }

        for bias in &mut network.hidden_biases {
            *bias = read_weight(
                &mut tokens,
                NETWORK_BIAS_LIMIT,
            )?;
        }

        for weight in &mut network.value_weights {
            *weight = read_weight(
                &mut tokens,
                NETWORK_WEIGHT_LIMIT,
            )? as i16;
        }

        for head in [&mut network.origin_weights, &mut network.destination_weights,] {
            for weight in head.iter_mut().flatten() {
                *weight = read_weight(
                    &mut tokens,
                    NETWORK_WEIGHT_LIMIT,
                )? as i16;
            }
        }

        for biases in [&mut network.origin_biases, &mut network.destination_biases,] {
            for bias in biases {
                *bias = read_weight(
                    &mut tokens,
                    NETWORK_WEIGHT_LIMIT,
                )? as i16;
            }
        }

        network.value_bias = read_weight(
            &mut tokens,
            NETWORK_BIAS_LIMIT,
        )?;

        if tokens.next().is_some() || (Some(network.get_fingerprint()) != expected_fingerprint) {
            return Err(EngineError::create_invalid_input("Failed to load the network because its checksum or payload is invalid."));
        }

        return Ok(network);
    }

    pub fn save_network(
        &self,
        path: &Path,
    ) -> EngineResult<()> {
        if let Some(network) = &self.stockfish_network {
            return network.save_network(path);
        }

        if let Some(network) = &self.neural_network {
            return network.save_network(path);
        }

        let mut output = BufWriter::new(File::create(path)?);
        writeln!(
            output,
            "ALEKHINE_NETWORK 1 {NETWORK_FEATURE_COUNT} {NETWORK_HIDDEN_COUNT} {} {}",
            u8::from(self.has_trained_weights),
            self.get_fingerprint(),
        )?;

        for weights in &self.feature_weights {
            for weight in weights {
                write!(
                    output,
                    "{weight} ",
                )?;
            }

            writeln!(output)?;
        }

        for bias in &self.hidden_biases {
            writeln!(
                output,
                "{bias}",
            )?;
        }

        for weight in &self.value_weights {
            writeln!(
                output,
                "{weight}",
            )?;
        }

        for head in [&self.origin_weights, &self.destination_weights,] {
            for weights in head {
                for weight in weights {
                    write!(
                        output,
                        "{weight} ",
                    )?;
                }

                writeln!(output)?;
            }
        }

        for biases in [&self.origin_biases, &self.destination_biases,] {
            for bias in biases {
                writeln!(
                    output,
                    "{bias}",
                )?;
            }
        }

        writeln!(
            output,
            "{}",
            self.value_bias,
        )?;
        output.flush()?;

        return Ok(());
    }
}

pub fn compute_outcome_probabilities(centipawn_score: i32) -> OutcomeProbabilities {
    let strength = centipawn_score as f64 / 240.0;
    let win = 1.0 / (1.0 + (1.0 - strength).exp());
    let loss = 1.0 / (1.0 + (1.0 + strength).exp());
    let draw = (1.0 - win - loss).max(0.0);
    let total = win + draw + loss;

    return OutcomeProbabilities {
        win: win / total,
        draw: draw / total,
        loss: loss / total,
    };
}

#[cfg(test)]
mod tests {
    use crate::evaluation::neural::{
        NEURAL_QUANTIZATION_SCALE,
        NeuralNetwork,
    };
    use crate::chess::position::Position;
    use crate::chess::types::START_FEN;
    use super::Network;

    #[test]
    fn test_policy_only_model_preserves_bootstrap_value_and_uses_learned_policy() {
        let mut neural = NeuralNetwork::create_zero_network();
        neural.has_trained_policy = true;
        neural.output_biases[4..].fill(NEURAL_QUANTIZATION_SCALE as i16);
        let model = Network::create_neural_network(neural);
        let bootstrap = Network::create_network();
        let mut position = Position::parse_fen(START_FEN).unwrap();
        let accumulator = model.build_accumulator(&position);
        assert!(!model.is_trained());
        assert_eq!(
            model.evaluate_position(
                &position,
                &accumulator,
            ),
            bootstrap.evaluate_position(
                &position,
                &bootstrap.build_accumulator(&position),
            ),
        );
        let chess_move = position.find_legal_move("e2e4").unwrap();
        assert_eq!(
            model.evaluate_policy(
                &position,
                &accumulator,
                chess_move,
            ),
            2.0,
        );
    }

    #[test]
    fn test_neural_value_bias_is_side_to_move_relative_without_bootstrap_tempo() {
        let mut neural = NeuralNetwork::create_zero_network();
        neural.has_trained_value = true;
        neural.output_biases[0] = NEURAL_QUANTIZATION_SCALE as i16;
        let model = Network::create_neural_network(neural);

        for side in ["w", "b",] {
            let position = Position::parse_fen(&format!("4k3/8/8/8/8/8/4P3/4K3 {side} - - 0 1")).unwrap();
            assert_eq!(
                model.evaluate_position(
                    &position,
                    &model.build_accumulator(&position),
                ),
                400,
            );
        }
    }

    #[test]
    fn test_outcome_head_matches_softmax_and_remains_finite_at_extreme_logits() {
        let position = Position::parse_fen(START_FEN).unwrap();

        for (biases, expected) in [([256, 0, 0,], (1.0f64.exp() - 1.0) / (1.0f64.exp() + 2.0),), ([0, 256, 0,], 0.0,), ([0, 0, 256,], (1.0 - 1.0f64.exp()) / (1.0f64.exp() + 2.0),), ([4096, 4096, 4096,], 0.0,),] {
            let mut neural = NeuralNetwork::create_zero_network();
            neural.has_trained_outcome = true;
            neural.output_biases[1..4].copy_from_slice(&biases);
            let model = Network::create_neural_network(neural);
            let outcome = model.evaluate_expected_outcome(
                &position,
                &model.build_accumulator(&position),
            );
            assert!(outcome.is_finite());
            assert!((outcome - expected).abs() < 0.000000001);
        }
    }

    #[test]
    fn test_outcome_fallback_uses_learned_value_when_outcome_head_is_untrained() {
        let mut neural = NeuralNetwork::create_zero_network();
        neural.has_trained_value = true;
        neural.output_biases[0] = NEURAL_QUANTIZATION_SCALE as i16;
        let model = Network::create_neural_network(neural);
        let position = Position::parse_fen(START_FEN).unwrap();
        assert!((model.evaluate_expected_outcome(
            &position,
            &model.build_accumulator(&position),
        ) - 1.0f64.tanh()).abs() < 0.000000001);
    }

    #[test]
    fn test_extreme_neural_values_stay_outside_reserved_mate_scores() {
        let position = Position::parse_fen(START_FEN).unwrap();

        for sign in [-1, 1,] {
            let mut neural = NeuralNetwork::create_zero_network();
            neural.has_trained_value = true;
            neural.hidden_biases.fill(4096);
            neural.output_weights.fill(sign * 4096);
            let model = Network::create_neural_network(neural);
            assert_eq!(
                model.evaluate_position(
                    &position,
                    &model.build_accumulator(&position),
                ),
                sign as i32 * 20000,
            );
        }
    }
}
