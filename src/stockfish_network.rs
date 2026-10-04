use std::fs::{
    File,
    OpenOptions,
};
use std::io::{
    BufWriter,
    Read,
    Write,
};
use std::path::Path;
use std::sync::{
    Arc,
    OnceLock,
};

use crate::fingerprint::{
    compute_byte_fingerprint,
    update_byte_fingerprint,
};
use crate::position::Position;
use crate::tactical::can_attack_square;
use crate::types::{
    BOARD_SQUARE_COUNT,
    Color,
    EngineError,
    EngineResult,
};

pub const STOCKFISH_VERSION: u32 = 0x7af32f20;
pub const STOCKFISH_BUCKET_COUNT: usize = 8;
pub const STOCKFISH_MAXIMUM_BYTES: u64 = 128 * 1024 * 1024;
pub(crate) const DENSE_FIRST_OUTPUT_COUNT: usize = 16;
pub(crate) const DENSE_SECOND_INPUT_COUNT: usize = 30;
pub(crate) const DENSE_SECOND_OUTPUT_COUNT: usize = 32;
pub(crate) const DENSE_INPUT_STRIDE: usize = 32;
pub(crate) const DENSE_WEIGHT_SCALE: f64 = 64.0;
pub(crate) const DENSE_BIAS_SCALE: f64 = 127.0 * DENSE_WEIGHT_SCALE;
pub(crate) const DENSE_OUTPUT_SCALE: f64 = DENSE_BIAS_SCALE / 16.0;
pub(crate) const DENSE_FORWARD_SCALE: f64 = 600.0;
const PIECE_FEATURE_COUNT: usize = 22528;
const THREAT_FEATURE_COUNT: usize = 79856;
const PIECE_BUCKET_WIDTH: usize = 704;
const BIG_NETWORK_HASH: u32 = 0xec102ef2;
const SMALL_NETWORK_HASH: u32 = 0x1c103c92;
const BIG_TRANSFORMER_HASH: u32 = 0x8f2344b8;
const SMALL_TRANSFORMER_HASH: u32 = 0x7f234db8;
const BIG_STACK_HASH: u32 = 0x63336a4a;
const SMALL_STACK_HASH: u32 = 0x6333712a;
const MAXIMUM_DESCRIPTION_BYTES: usize = 4096;
const MAXIMUM_DENSE_BIAS: i32 = 100_000_000;
const MAXIMUM_PSQT_WEIGHT: u32 = 1_000_000;
const LEB_MAGIC: &[u8] = b"COMPRESSED_LEB128";
const MAXIMUM_HIDDEN_WIDTH: usize = 1024;
static THREAT_GEOMETRY: OnceLock<ThreatGeometry> = OnceLock::new();

struct ByteCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

fn create_format_error() -> EngineError {
    return EngineError::create_invalid_input("Failed to read a Stockfish 18 network because its format, dimensions, or parameters are invalid.");
}

impl<'a> ByteCursor<'a> {
    fn read_bytes(
        &mut self,
        count: usize,
    ) -> EngineResult<&'a [u8]> {
        let end = self.offset.checked_add(count).ok_or_else(create_format_error)?;
        let bytes = self.bytes.get(self.offset..end).ok_or_else(create_format_error)?;
        self.offset = end;

        return Ok(bytes);
    }

    fn read_unsigned(&mut self) -> EngineResult<u32> {
        return Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().map_err(|_| create_format_error())?));
    }

    fn read_compressed<T>(
        &mut self,
        count: usize,
        maximum_byte_count: usize,
    ) -> EngineResult<Vec<T>>
    where
        T: TryFrom<i64>,
    {
        if self.read_bytes(LEB_MAGIC.len())? != LEB_MAGIC {
            return Err(create_format_error());
        }

        let byte_count = self.read_unsigned()? as usize;

        if (byte_count < count) || (byte_count > count * maximum_byte_count) {
            return Err(create_format_error());
        }

        let compressed = self.read_bytes(byte_count)?;
        let mut parameters = Vec::with_capacity(count);
        let mut offset = 0;

        for _ in 0..count {
            let mut parameter = 0_i64;
            let mut was_terminated = false;

            for index in 0..maximum_byte_count {
                let byte = *compressed.get(offset).ok_or_else(create_format_error)?;
                offset += 1;
                parameter |= i64::from(byte & 127) << (index * 7);

                if byte & 128 == 0 {
                    if byte & 64 != 0 {
                        parameter |= -1_i64 << ((index + 1) * 7);
                    }

                    was_terminated = true;

                    break;
                }
            }

            if !was_terminated {
                return Err(create_format_error());
            }

            parameters.push(T::try_from(parameter).map_err(|_| create_format_error())?);
        }

        if offset != compressed.len() {
            return Err(create_format_error());
        }

        return Ok(parameters);
    }
}

#[derive(Clone)]
pub(crate) struct DenseLayer {
    pub(crate) biases: Vec<i32>,
    pub(crate) weights: Vec<i8>,
    pub(crate) input_count: usize,
}

impl DenseLayer {
    fn read_layer(
        cursor: &mut ByteCursor<'_>,
        input_count: usize,
        output_count: usize,
    ) -> EngineResult<Self> {
        let mut biases = Vec::with_capacity(output_count);

        for _ in 0..output_count {
            let bias = i32::from_le_bytes(cursor.read_bytes(4)?.try_into().map_err(|_| create_format_error())?);

            if bias.unsigned_abs() > MAXIMUM_DENSE_BIAS as u32 {
                return Err(create_format_error());
            }

            biases.push(bias);
        }

        let weights = cursor.read_bytes(input_count * output_count)?.iter().map(|&weight| weight as i8).collect();

        return Ok(Self {
            biases,
            weights,
            input_count,
        });
    }

    fn calculate_outputs(
        &self,
        inputs: &[u8],
    ) -> Vec<i32> {
        return self.biases.iter().enumerate().map(|(output_index, &bias)| {
            let weights = &self.weights[output_index * self.input_count..(output_index + 1) * self.input_count];
            let sum = weights.iter().zip(inputs).fold(
                i64::from(bias),
                |sum, (&weight, &input)| sum + i64::from(weight) * i64::from(input),
            );

            return sum as i32;
        }).collect();
    }

    fn append_bytes(
        &self,
        bytes: &mut Vec<u8>,
    ) {
        for bias in &self.biases {
            bytes.extend_from_slice(&bias.to_le_bytes());
        }

        bytes.extend(self.weights.iter().map(|&weight| weight as u8));
    }
}

#[derive(Clone)]
pub(crate) struct DenseStack {
    pub(crate) layers: [DenseLayer; 3],
}

struct FeatureTransformer {
    biases: Vec<i16>,
    piece_weights: Vec<i16>,
    threat_weights: Vec<i8>,
    psqt_weights: Vec<i32>,
}

#[derive(Clone)]
pub struct StockfishNetwork {
    source_bytes: Arc<Vec<u8>>,
    transformer: Arc<FeatureTransformer>,
    dense_offset: usize,
    prefix_fingerprint: u64,
    hidden_width: usize,
    pub(crate) stacks: [DenseStack; STOCKFISH_BUCKET_COUNT],
}

pub struct StockfishFeatures {
    pub inputs: Vec<u8>,
    pub psqt_scores: [i32; STOCKFISH_BUCKET_COUNT],
    pub bucket_index: usize,
    pub centipawn_scale: f64,
}

#[derive(
    Clone,
    Copy,
)]
pub struct StockfishEvaluation {
    pub psqt_score: i32,
    pub positional_score: i32,
    pub centipawn_score: i32,
}

struct ThreatGeometry {
    attacks: [[u64; BOARD_SQUARE_COUNT]; 12],
    square_offsets: [[usize; BOARD_SQUARE_COUNT]; 12],
    edge_counts: [usize; 12],
    piece_offsets: [usize; 12],
}

fn get_target_types(piece_type: usize) -> &'static [usize] {
    match piece_type {
        1 => return &[1, 2, 4,],
        3 | 4 => return &[1, 2, 3, 4, 6,],
        6 => return &[1, 2, 3, 4,],
        _ => return &[1, 2, 3, 4, 5, 6,],
    }
}

fn build_threat_geometry() -> ThreatGeometry {
    let mut geometry = ThreatGeometry {
        attacks: [[0; BOARD_SQUARE_COUNT]; 12],
        square_offsets: [[0; BOARD_SQUARE_COUNT]; 12],
        edge_counts: [0; 12],
        piece_offsets: [0; 12],
    };
    let mut total_feature_count = 0;

    for piece_index in 0..12 {
        let piece_type = piece_index % 6 + 1;
        let sign = if piece_index < 6 { 1 } else { -1 };
        geometry.piece_offsets[piece_index] = total_feature_count;

        for origin in 0..BOARD_SQUARE_COUNT {
            geometry.square_offsets[piece_index][origin] = geometry.edge_counts[piece_index];
            let mut board = [0; BOARD_SQUARE_COUNT];
            board[origin] = sign * piece_type as i32;

            if (piece_type == 1) && !(8..56).contains(&origin) {
                continue;
            }

            for destination in 0..BOARD_SQUARE_COUNT {
                if can_attack_square(
                    &board,
                    origin,
                    destination,
                ) {
                    geometry.attacks[piece_index][origin] |= 1_u64 << destination;
                }
            }

            geometry.edge_counts[piece_index] += geometry.attacks[piece_index][origin].count_ones() as usize;
        }

        total_feature_count += geometry.edge_counts[piece_index] * get_target_types(piece_type).len() * 2;
    }

    assert_eq!(
        total_feature_count,
        THREAT_FEATURE_COUNT,
    );

    return geometry;
}

fn get_king_square(
    board: &[i32; BOARD_SQUARE_COUNT],
    perspective: usize,
) -> usize {
    let king = if perspective == 0 { 6 } else { -6 };

    return board.iter().position(|&piece| piece == king).expect("A validated position must contain both kings.");
}

fn encode_piece_features(
    board: &[i32; BOARD_SQUARE_COUNT],
    perspective: usize,
) -> Vec<usize> {
    let king_square = get_king_square(
        board,
        perspective,
    );
    let rank_orientation = perspective * 56;
    let file_orientation = if king_square % 8 < 4 { 7 } else { 0 };
    let oriented_king = king_square ^ rank_orientation;
    let king_bucket = (7 - oriented_king / 8) * 4 + (oriented_king % 8).min(7 - oriented_king % 8);

    return board.iter().enumerate().filter_map(|(square, &piece)| {
        if piece == 0 {
            return None;
        }

        let piece_type = piece.unsigned_abs() as usize;
        let is_enemy = (piece < 0) != (perspective == 1);
        let piece_channel = if piece_type == 6 { 10 } else { (piece_type - 1) * 2 + usize::from(is_enemy) };

        return Some(king_bucket * PIECE_BUCKET_WIDTH + piece_channel * BOARD_SQUARE_COUNT + (square ^ rank_orientation ^ file_orientation));
    }).collect();
}

fn encode_threat_features(
    board: &[i32; BOARD_SQUARE_COUNT],
    perspective: usize,
) -> Vec<usize> {
    let geometry = THREAT_GEOMETRY.get_or_init(build_threat_geometry);
    let king_square = get_king_square(
        board,
        perspective,
    );
    let orientation = perspective * 56 + if king_square % 8 >= 4 { 7 } else { 0 };
    let mut features = Vec::new();

    for (origin, &attacker,) in board.iter().enumerate() {
        if attacker == 0 {
            continue;
        }

        let attacker_type = attacker.unsigned_abs() as usize;
        let attacker_color = usize::from((attacker < 0) != (perspective == 1));
        let piece_index = attacker_color * 6 + attacker_type - 1;
        let target_types = get_target_types(attacker_type);
        let oriented_origin = origin ^ orientation;

        for (destination, &target,) in board.iter().enumerate() {
            if (target == 0) || !can_attack_square(
                board,
                origin,
                destination,
            ) {
                continue;
            }

            let target_type = target.unsigned_abs() as usize;
            let Some(target_type_index) = target_types.iter().position(|&piece_type| piece_type == target_type) else {
                continue;
            };
            let oriented_destination = destination ^ orientation;
            let has_redundant_direction = (attacker_type == target_type)
                && (((attacker < 0) != (target < 0)) || (attacker_type != 1))
                && (oriented_origin < oriented_destination);

            if has_redundant_direction {
                continue;
            }

            let target_color = usize::from((target < 0) != (perspective == 1));
            let target_channel = target_color * target_types.len() + target_type_index;
            let preceding_squares = (1_u64 << oriented_destination).wrapping_sub(1);
            let destination_index = (geometry.attacks[piece_index][oriented_origin] & preceding_squares).count_ones() as usize;
            features.push(geometry.piece_offsets[piece_index]
                + target_channel * geometry.edge_counts[piece_index]
                + geometry.square_offsets[piece_index][oriented_origin]
                + destination_index);
        }
    }

    return features;
}

pub fn calculate_stockfish_centipawn_scale(board: &[i32; BOARD_SQUARE_COUNT]) -> f64 {
    let material_weights = [0, 1, 3, 3, 5, 9, 0,];
    let material: i32 = board.iter().map(|piece| material_weights[piece.unsigned_abs() as usize]).sum();
    let normalized_material = material.clamp(
        17,
        78,
    ) as f64 / 58.0;
    let normalization = -72.32565836 * normalized_material.powi(3)
        + 185.93832038 * normalized_material.powi(2)
        - 144.58862193 * normalized_material
        + 416.44950446;

    return 100.0 / normalization;
}

impl StockfishNetwork {
    pub fn load_network(path: &Path) -> EngineResult<Self> {
        let file = File::open(path)?;

        if file.metadata()?.len() > STOCKFISH_MAXIMUM_BYTES {
            return Err(create_format_error());
        }

        let mut bytes = Vec::new();
        file.take(STOCKFISH_MAXIMUM_BYTES + 1).read_to_end(&mut bytes)?;

        return Self::decode_network(bytes);
    }

    pub fn decode_network(bytes: Vec<u8>) -> EngineResult<Self> {
        if bytes.len() as u64 > STOCKFISH_MAXIMUM_BYTES {
            return Err(create_format_error());
        }

        let mut cursor = ByteCursor {
            bytes: &bytes,
            offset: 0,
        };

        if cursor.read_unsigned()? != STOCKFISH_VERSION {
            return Err(create_format_error());
        }

        let network_hash = cursor.read_unsigned()?;
        let hidden_width = match network_hash {
            BIG_NETWORK_HASH => 1024,
            SMALL_NETWORK_HASH => 128,
            _ => return Err(create_format_error()),
        };
        let description_length = cursor.read_unsigned()? as usize;

        if description_length > MAXIMUM_DESCRIPTION_BYTES {
            return Err(create_format_error());
        }

        cursor.read_bytes(description_length)?;
        let is_big = hidden_width == MAXIMUM_HIDDEN_WIDTH;
        let transformer_hash = if is_big { BIG_TRANSFORMER_HASH } else { SMALL_TRANSFORMER_HASH };

        if cursor.read_unsigned()? != transformer_hash {
            return Err(create_format_error());
        }

        let biases = cursor.read_compressed::<i16>(
            hidden_width,
            3,
        )?;
        let threat_weights = if is_big {
            cursor.read_bytes(THREAT_FEATURE_COUNT * hidden_width)?.iter().map(|&weight| weight as i8).collect()
        } else {
            Vec::new()
        };
        let piece_weights = cursor.read_compressed::<i16>(
            PIECE_FEATURE_COUNT * hidden_width,
            3,
        )?;
        let psqt_count = (PIECE_FEATURE_COUNT + if is_big { THREAT_FEATURE_COUNT } else { 0 }) * STOCKFISH_BUCKET_COUNT;
        let psqt_weights = cursor.read_compressed::<i32>(
            psqt_count,
            5,
        )?;

        if psqt_weights.iter().any(|parameter| parameter.unsigned_abs() > MAXIMUM_PSQT_WEIGHT) {
            return Err(create_format_error());
        }

        let dense_offset = cursor.offset;
        let mut stacks = Vec::new();

        for _ in 0..STOCKFISH_BUCKET_COUNT {
            if cursor.read_unsigned()? != if is_big { BIG_STACK_HASH } else { SMALL_STACK_HASH } {
                return Err(create_format_error());
            }

            stacks.push(DenseStack {
                layers: [
                    DenseLayer::read_layer(
                        &mut cursor,
                        hidden_width,
                        DENSE_FIRST_OUTPUT_COUNT,
                    )?,
                    DenseLayer::read_layer(
                        &mut cursor,
                        DENSE_INPUT_STRIDE,
                        DENSE_SECOND_OUTPUT_COUNT,
                    )?,
                    DenseLayer::read_layer(
                        &mut cursor,
                        DENSE_INPUT_STRIDE,
                        1,
                    )?,
                ],
            });
        }

        if cursor.offset != bytes.len() {
            return Err(create_format_error());
        }

        let prefix_fingerprint = compute_byte_fingerprint(&bytes[..dense_offset]);

        return Ok(Self {
            source_bytes: Arc::new(bytes),
            transformer: Arc::new(FeatureTransformer {
                biases,
                piece_weights,
                threat_weights,
                psqt_weights,
            }),
            dense_offset,
            prefix_fingerprint,
            hidden_width,
            stacks: stacks.try_into().map_err(|_| create_format_error())?,
        });
    }

    pub fn get_hidden_width(&self) -> usize {
        return self.hidden_width;
    }

    pub fn get_frozen_bytes(&self) -> &[u8] {
        return &self.source_bytes[..self.dense_offset];
    }

    fn encode_dense_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let stack_hash = if self.hidden_width == MAXIMUM_HIDDEN_WIDTH { BIG_STACK_HASH } else { SMALL_STACK_HASH };

        for stack in &self.stacks {
            bytes.extend_from_slice(&stack_hash.to_le_bytes());

            for layer in &stack.layers {
                layer.append_bytes(&mut bytes);
            }
        }

        return bytes;
    }

    pub fn get_fingerprint(&self) -> u64 {
        let mut fingerprint = self.prefix_fingerprint;
        update_byte_fingerprint(
            &mut fingerprint,
            &self.encode_dense_bytes(),
        );

        return fingerprint;
    }

    pub fn save_network(
        &self,
        path: &Path,
    ) -> EngineResult<()> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut output = BufWriter::new(file);
        output.write_all(self.get_frozen_bytes())?;
        output.write_all(&self.encode_dense_bytes())?;
        output.flush()?;

        return Ok(());
    }

    pub fn build_features(
        &self,
        board: &[i32; BOARD_SQUARE_COUNT],
        side: Color,
    ) -> StockfishFeatures {
        let is_big = self.hidden_width == MAXIMUM_HIDDEN_WIDTH;
        let mut sums = [[0_i16; MAXIMUM_HIDDEN_WIDTH]; 2];
        let mut psqt_sums = [[0_i64; STOCKFISH_BUCKET_COUNT]; 2];

        for perspective in 0..2 {
            sums[perspective][..self.hidden_width].copy_from_slice(&self.transformer.biases);
            let piece_features = encode_piece_features(
                board,
                perspective,
            );

            for feature in piece_features {
                let weights = &self.transformer.piece_weights[feature * self.hidden_width..(feature + 1) * self.hidden_width];

                for (activation, &weight,) in sums[perspective][..self.hidden_width].iter_mut().zip(weights) {
                    *activation = activation.wrapping_add(weight);
                }

                let psqt_feature = feature + if is_big { THREAT_FEATURE_COUNT } else { 0 };

                for (bucket, sum,) in psqt_sums[perspective].iter_mut().enumerate() {
                    *sum += i64::from(self.transformer.psqt_weights[psqt_feature * STOCKFISH_BUCKET_COUNT + bucket]);
                }
            }

            if is_big {
                for feature in encode_threat_features(
                    board,
                    perspective,
                ) {
                    let weights = &self.transformer.threat_weights[feature * self.hidden_width..(feature + 1) * self.hidden_width];

                    for (activation, &weight,) in sums[perspective][..self.hidden_width].iter_mut().zip(weights) {
                        *activation = activation.wrapping_add(i16::from(weight));
                    }

                    for (bucket, sum,) in psqt_sums[perspective].iter_mut().enumerate() {
                        *sum += i64::from(self.transformer.psqt_weights[feature * STOCKFISH_BUCKET_COUNT + bucket]);
                    }
                }
            } else {
                for activation in &mut sums[perspective][..self.hidden_width] {
                    *activation = activation.wrapping_mul(2);
                }
            }
        }

        let moving_side = usize::from(side == Color::BLACK);
        let mut inputs = Vec::with_capacity(self.hidden_width);
        let activation_limit = if is_big { 255 } else { 254 };

        for perspective in [moving_side, moving_side ^ 1,] {
            for channel in 0..self.hidden_width / 2 {
                let first = i32::from(sums[perspective][channel]).clamp(
                    0,
                    activation_limit,
                );
                let second = i32::from(sums[perspective][channel + self.hidden_width / 2]).clamp(
                    0,
                    activation_limit,
                );
                inputs.push(((first * second) / 512) as u8);
            }
        }

        let piece_count = board.iter().filter(|&&piece| piece != 0).count();
        let psqt_scores = std::array::from_fn(|bucket| ((psqt_sums[moving_side][bucket] - psqt_sums[moving_side ^ 1][bucket]) / 2 / 16) as i32);

        return StockfishFeatures {
            inputs,
            psqt_scores,
            bucket_index: (piece_count.saturating_sub(1) / 4).min(STOCKFISH_BUCKET_COUNT - 1),
            centipawn_scale: calculate_stockfish_centipawn_scale(board),
        };
    }

    pub fn evaluate_features(
        &self,
        features: &StockfishFeatures,
        bucket_index: usize,
    ) -> StockfishEvaluation {
        let stack = &self.stacks[bucket_index];
        let first_outputs = stack.layers[0].calculate_outputs(&features.inputs);
        let mut second_inputs = [0; DENSE_INPUT_STRIDE];

        for channel in 0..DENSE_FIRST_OUTPUT_COUNT - 1 {
            let activation = i64::from(first_outputs[channel]);
            second_inputs[channel] = ((activation * activation) >> 19).min(127) as u8;
            second_inputs[channel + DENSE_FIRST_OUTPUT_COUNT - 1] = (first_outputs[channel] >> 6).clamp(
                0,
                127,
            ) as u8;
        }

        let second_outputs = stack.layers[1].calculate_outputs(&second_inputs);
        let final_inputs: Vec<u8> = second_outputs.iter().map(|&activation| (activation >> 6).clamp(
            0,
            127,
        ) as u8).collect();
        let output = stack.layers[2].calculate_outputs(&final_inputs)[0];
        let forward_score = i64::from(first_outputs[DENSE_FIRST_OUTPUT_COUNT - 1]) * 9600 / 8128;
        let positional_score = ((i64::from(output) + forward_score) / 16) as i32;
        let psqt_score = features.psqt_scores[bucket_index];
        let centipawn_score = ((f64::from(psqt_score) + f64::from(positional_score)) * features.centipawn_scale).round().clamp(
            -20000.0,
            20000.0,
        ) as i32;

        return StockfishEvaluation {
            psqt_score,
            positional_score,
            centipawn_score,
        };
    }

    pub fn evaluate_position(
        &self,
        position: &Position,
    ) -> i32 {
        let board = std::array::from_fn(|square| position.get_piece(square as i32));
        let features = self.build_features(
            &board,
            position.get_side_to_move(),
        );

        return self.evaluate_features(
            &features,
            features.bucket_index,
        ).centipawn_score;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ByteCursor,
        DenseLayer,
        LEB_MAGIC,
    };

    fn create_compressed_bytes(payload: &[u8]) -> Vec<u8> {
        let mut bytes = LEB_MAGIC.to_vec();
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(payload);

        return bytes;
    }

    #[test]
    fn test_signed_compression_decodes_boundaries() {
        let bytes = create_compressed_bytes(&[0x80, 0x80, 0x7e, 0x7f, 0x00, 0xff, 0xff, 0x01,]);
        let mut cursor = ByteCursor {
            bytes: &bytes,
            offset: 0,
        };
        assert_eq!(
            cursor.read_compressed::<i16>(
                4,
                3,
            ).unwrap(),
            vec![i16::MIN, -1, 0, i16::MAX,],
        );
    }

    #[test]
    fn test_signed_compression_rejects_overflow_and_unterminated_values() {
        for payload in [&[0x80, 0x80, 0x02,][..], &[0x80, 0x80, 0x80,][..], &[0x80,][..],] {
            let bytes = create_compressed_bytes(payload);
            let mut cursor = ByteCursor {
                bytes: &bytes,
                offset: 0,
            };
            assert!(cursor.read_compressed::<i16>(
                1,
                3,
            ).is_err());
        }
    }

    #[test]
    fn test_signed_compression_rejects_unconsumed_and_truncated_payloads() {
        for payload in [&[0, 0,][..], &[0x80,][..],] {
            let bytes = create_compressed_bytes(payload);
            let mut cursor = ByteCursor {
                bytes: &bytes,
                offset: 0,
            };
            assert!(cursor.read_compressed::<i16>(
                1,
                3,
            ).is_err());
        }
    }

    #[test]
    fn test_dense_affine_uses_signed_weights_and_wide_accumulation() {
        let layer = DenseLayer {
            biases: vec![-100, 100,],
            weights: vec![-128, 127, 127, -128,],
            input_count: 2,
        };
        assert_eq!(
            layer.calculate_outputs(&[127, 100,]),
            vec![-3656, 3429,],
        );
    }
}
