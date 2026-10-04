use std::collections::{
    HashMap,
    HashSet,
};
use std::fs::{
    File,
    OpenOptions,
};
use std::io::{
    BufRead,
    BufReader,
    Read,
    Write,
};
use std::path::Path;

use crate::fingerprint::{
    compute_byte_fingerprint,
    FINGERPRINT_OFFSET,
    update_byte_fingerprint,
};
use crate::neural::{
    encode_features,
    FeatureSet,
    get_policy_head_indices,
    NEURAL_FEATURE_COUNT,
    NEURAL_HEAD_COUNT,
    NEURAL_HIDDEN_COUNT,
    NEURAL_INPUT_COUNT,
    NEURAL_QUANTIZATION_SCALE,
    NEURAL_SCORE_SCALE,
    NeuralNetwork,
};
use crate::position::Position;
use crate::types::{
    BOARD_SQUARE_COUNT,
    Color,
    EngineError,
    EngineResult,
};

const FEATURE_PARAMETER_COUNT: usize = NEURAL_FEATURE_COUNT * NEURAL_HIDDEN_COUNT;
const OUTPUT_PARAMETER_OFFSET: usize = FEATURE_PARAMETER_COUNT + NEURAL_HIDDEN_COUNT;
const OUTPUT_BIAS_OFFSET: usize = OUTPUT_PARAMETER_OFFSET + NEURAL_HEAD_COUNT * NEURAL_INPUT_COUNT;
const PARAMETER_COUNT: usize = OUTPUT_BIAS_OFFSET + NEURAL_HEAD_COUNT;
const MAXIMUM_RECORD_BYTES: usize = 8192;
pub(crate) const ADAM_FIRST_DECAY: f32 = 0.9;
pub(crate) const ADAM_SECOND_DECAY: f32 = 0.999;
pub(crate) const ADAM_EPSILON: f32 = 0.00000001;
pub(crate) const WEIGHT_DECAY: f32 = 0.0001;
const OUTCOME_MIXTURE: f32 = 0.25;

#[derive(Clone)]
pub struct TrainingOptions {
    pub epoch_count: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub seed: u64,
    pub maximum_record_count: usize,
}

impl Default for TrainingOptions {
    fn default() -> Self {
        return Self {
            epoch_count: 10,
            batch_size: 256,
            learning_rate: 0.001,
            seed: 20261004,
            maximum_record_count: 100000,
        };
    }
}

pub struct TrainingReport {
    pub training_record_count: usize,
    pub validation_record_count: usize,
    pub duplicate_record_count: usize,
    pub initial_validation_loss: f32,
    pub best_validation_loss: f32,
    pub maximum_quantization_error_centipawns: f32,
    pub network_fingerprint: u64,
    pub dataset_fingerprint: u64,
}

struct PolicyTarget {
    heads: [usize; 3],
    probability: f32,
}

pub(crate) struct TrainingRecord {
    pub(crate) board: [i32; BOARD_SQUARE_COUNT],
    pub(crate) side: Color,
    pub(crate) value_target: f32,
    outcome_target: Option<[f32; 3]>,
    policies: Vec<PolicyTarget>,
    pub(crate) is_validation: bool,
}

#[derive(Clone)]
struct FloatingNetwork {
    parameters: Vec<f32>,
}

struct ForwardPass {
    features: FeatureSet,
    hidden: [f32; NEURAL_INPUT_COUNT],
    activations: [f32; NEURAL_INPUT_COUNT],
    outputs: [f32; NEURAL_HEAD_COUNT],
}

pub(crate) struct RandomSequence {
    state: u64,
}

impl RandomSequence {
    pub(crate) fn create_sequence(seed: u64) -> Self {
        return Self {
            state: seed.max(1),
        };
    }

    pub(crate) fn generate_integer(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;

        return self.state;
    }

    fn generate_weight(&mut self) -> f32 {
        return ((self.generate_integer() >> 40) as f32 / 16777216.0 - 0.5) * 0.1;
    }
}

fn compute_group_hash(name: &str) -> u64 {
    return compute_byte_fingerprint(name.as_bytes());
}

pub(crate) fn compute_sigmoid(logit: f32) -> f32 {
    let bounded_logit = logit.clamp(
        -40.0,
        40.0,
    );

    return 1.0 / (1.0 + (-bounded_logit).exp());
}

pub(crate) fn compute_cross_entropy(
    prediction: f32,
    target: f32,
) -> f32 {
    let bounded_prediction = prediction.clamp(
        0.0000001,
        0.9999999,
    );

    return -(target * bounded_prediction.ln() + (1.0 - target) * (1.0 - bounded_prediction).ln());
}

fn parse_training_record(
    text: &str,
    seen_positions: &mut HashSet<String>,
) -> EngineResult<Option<TrainingRecord>> {
    let fields: Vec<&str> = text.split('|').map(str::trim).collect();

    if (fields.len() != 5) || fields[4].is_empty() {
        return Err(EngineError::create_invalid_input("Failed to parse a training record; expected FEN, White score, White outcome or ?, move probabilities or ?, and game identifier."));
    }

    let mut position = Position::parse_fen(fields[0])?;
    let white_score = fields[1].parse::<i32>().map_err(|_| EngineError::create_invalid_input("Failed to parse a training score."))?;

    if white_score.unsigned_abs() > 20000 {
        return Err(EngineError::create_invalid_input("Failed to parse a training score because mate and extreme scores require separate labels."));
    }

    let white_outcome = if fields[2] == "?" {
        None
    } else {
        let outcome = fields[2].parse::<f32>().map_err(|_| EngineError::create_invalid_input("Failed to parse a training outcome."))?;

        if ![0.0, 0.5, 1.0,].contains(&outcome) {
            return Err(EngineError::create_invalid_input("Failed to parse a training outcome; use 0, 0.5, 1, or ?."));
        }

        Some(outcome)
    };
    let side = position.get_side_to_move();
    let sign = if side == Color::WHITE {
        1.0
    } else {
        -1.0
    };
    let mut board = [0; BOARD_SQUARE_COUNT];

    for (square, piece) in board.iter_mut().enumerate() {
        *piece = position.get_piece(square as i32);
    }

    let outcome_target = white_outcome.map(|outcome| {
        let side_outcome = if side == Color::WHITE {
            outcome
        } else {
            1.0 - outcome
        };

        if side_outcome == 1.0 {
            return [1.0, 0.0, 0.0,];
        }

        if side_outcome == 0.0 {
            return [0.0, 0.0, 1.0,];
        }

        return [0.0, 1.0, 0.0,];
    });
    let teacher_probability = compute_sigmoid(white_score as f32 * sign / NEURAL_SCORE_SCALE as f32);
    let value_target = if let Some(outcome) = outcome_target {
        teacher_probability * (1.0 - OUTCOME_MIXTURE)
            + (outcome[0] + outcome[1] * 0.5) * OUTCOME_MIXTURE
    } else {
        teacher_probability
    };
    let mut policies = Vec::new();

    if fields[3] != "?" {
        let mut move_probabilities = HashMap::new();
        let mut probability_sum = 0.0;

        for target in fields[3].split_whitespace() {
            let (move_name, probability_text) = target.split_once(':')
                .ok_or_else(|| EngineError::create_invalid_input("Failed to parse a policy target; expected move:probability."))?;
            position.find_legal_move(move_name)?;
            let probability = probability_text.parse::<f32>()
                .map_err(|_| EngineError::create_invalid_input("Failed to parse a policy probability."))?;

            let has_duplicate_move = move_probabilities.insert(
                move_name.to_owned(),
                probability,
            ).is_some();

            if !probability.is_finite() || (probability < 0.0) || has_duplicate_move {
                return Err(EngineError::create_invalid_input("Failed to parse a policy target because its probability or move is invalid."));
            }

            probability_sum += probability;
        }

        if !probability_sum.is_finite() || (probability_sum <= 0.0) {
            return Err(EngineError::create_invalid_input("Failed to parse policy targets because their total probability is invalid."));
        }

        let legal_moves = position.generate_legal_moves(false)?;

        for &chess_move in &legal_moves.moves[..legal_moves.count] {
            policies.push(PolicyTarget {
                heads: get_policy_head_indices(
                    chess_move,
                    &board,
                    side,
                ),
                probability: move_probabilities.get(&chess_move.get_uci()).copied().unwrap_or(0.0) / probability_sum,
            });
        }
    }

    let position_key = fields[0].split_whitespace().take(4).collect::<Vec<&str>>().join(" ");

    if !seen_positions.insert(position_key) {
        return Ok(None);
    }

    return Ok(Some(TrainingRecord {
        board,
        side,
        value_target,
        outcome_target,
        policies,
        is_validation: compute_group_hash(fields[4]) % 5 == 0,
    }));
}

impl FloatingNetwork {
    fn create_network(random: &mut RandomSequence) -> Self {
        let mut parameters = vec![0.0; PARAMETER_COUNT];

        for parameter in &mut parameters[..FEATURE_PARAMETER_COUNT] {
            *parameter = random.generate_weight();
        }

        parameters[FEATURE_PARAMETER_COUNT..OUTPUT_PARAMETER_OFFSET].fill(0.25);

        for parameter in &mut parameters[OUTPUT_PARAMETER_OFFSET..OUTPUT_BIAS_OFFSET] {
            *parameter = random.generate_weight();
        }

        return Self {
            parameters,
        };
    }

    fn compute_forward_pass(
        &self,
        record: &TrainingRecord,
    ) -> ForwardPass {
        let features = encode_features(&record.board);
        let mut hidden = [0.0; NEURAL_INPUT_COUNT];
        let first_perspective = usize::from(record.side == Color::BLACK);

        for (input, hidden_activation) in hidden.iter_mut().enumerate() {
            let perspective = first_perspective ^ (input / NEURAL_HIDDEN_COUNT);
            let channel = input % NEURAL_HIDDEN_COUNT;
            *hidden_activation = self.parameters[FEATURE_PARAMETER_COUNT + channel];

            for feature in &features.features[perspective][..features.counts[perspective]] {
                *hidden_activation += self.parameters[feature.index as usize * NEURAL_HIDDEN_COUNT + channel] * feature.count as f32;
            }
        }

        let activations = hidden.map(|activation| activation.clamp(
            0.0,
            1.0,
        ).powi(2));
        let mut outputs = [0.0; NEURAL_HEAD_COUNT];

        for (head, output) in outputs.iter_mut().enumerate() {
            *output = self.parameters[OUTPUT_BIAS_OFFSET + head];

            for (input, activation) in activations.iter().enumerate() {
                *output += activation * self.parameters[OUTPUT_PARAMETER_OFFSET + head * NEURAL_INPUT_COUNT + input];
            }
        }

        return ForwardPass {
            features,
            hidden,
            activations,
            outputs,
        };
    }

    fn compute_loss_gradient(
        &self,
        record: &TrainingRecord,
        gradients: Option<&mut [f32]>,
    ) -> f32 {
        let forward = self.compute_forward_pass(record);
        let probability = compute_sigmoid(forward.outputs[0]);
        let mut loss = compute_cross_entropy(
            probability,
            record.value_target,
        );
        let mut output_gradients = [0.0; NEURAL_HEAD_COUNT];
        output_gradients[0] = probability - record.value_target;

        if let Some(targets) = record.outcome_target {
            let maximum = forward.outputs[1..4].iter().copied().fold(
                f32::NEG_INFINITY,
                f32::max,
            );
            let mut probabilities = [0.0; 3];

            for (index, probability) in probabilities.iter_mut().enumerate() {
                *probability = (forward.outputs[index + 1] - maximum).exp();
            }

            let total: f32 = probabilities.iter().sum();

            for (index, probability) in probabilities.iter_mut().enumerate() {
                *probability /= total;
                output_gradients[index + 1] = *probability - targets[index];
                loss -= targets[index] * probability.max(0.0000001).ln();
            }
        }

        if !record.policies.is_empty() {
            let logits: Vec<f32> = record.policies.iter().map(|policy| {
                let mut logit = forward.outputs[policy.heads[0]] + forward.outputs[policy.heads[1]];

                if policy.heads[2] != 0 {
                    logit += forward.outputs[policy.heads[2]];
                }

                return logit;
            }).collect();
            let maximum = logits.iter().copied().fold(
                f32::NEG_INFINITY,
                f32::max,
            );
            let total: f32 = logits.iter().map(|logit| (logit - maximum).exp()).sum();

            for (policy, logit) in record.policies.iter().zip(logits) {
                let probability = (logit - maximum).exp() / total;
                let gradient = probability - policy.probability;
                output_gradients[policy.heads[0]] += gradient;
                output_gradients[policy.heads[1]] += gradient;

                if policy.heads[2] != 0 {
                    output_gradients[policy.heads[2]] += gradient;
                }

                loss -= policy.probability * probability.max(0.0000001).ln();
            }
        }

        let Some(gradients) = gradients else {
            return loss;
        };
        let mut hidden_gradients = [0.0; NEURAL_INPUT_COUNT];

        for (head, &output_gradient) in output_gradients.iter().enumerate() {
            if output_gradient == 0.0 {
                continue;
            }

            gradients[OUTPUT_BIAS_OFFSET + head] += output_gradient;

            for (input, &activation) in forward.activations.iter().enumerate() {
                let parameter_index = OUTPUT_PARAMETER_OFFSET + head * NEURAL_INPUT_COUNT + input;
                gradients[parameter_index] += output_gradient * activation;
                hidden_gradients[input] += output_gradient * self.parameters[parameter_index];
            }
        }

        let first_perspective = usize::from(record.side == Color::BLACK);

        for (input, &hidden_gradient) in hidden_gradients.iter().enumerate() {
            if (forward.hidden[input] <= 0.0) || (forward.hidden[input] >= 1.0) {
                continue;
            }

            let gradient = hidden_gradient * 2.0 * forward.hidden[input];
            let channel = input % NEURAL_HIDDEN_COUNT;
            let perspective = first_perspective ^ (input / NEURAL_HIDDEN_COUNT);
            gradients[FEATURE_PARAMETER_COUNT + channel] += gradient;

            for feature in &forward.features.features[perspective][..forward.features.counts[perspective]] {
                gradients[feature.index as usize * NEURAL_HIDDEN_COUNT + channel] += gradient * feature.count as f32;
            }
        }

        return loss;
    }

    fn quantize_network(
        &self,
        has_trained_policy: bool,
        has_trained_outcome: bool,
    ) -> NeuralNetwork {
        let weights: Vec<i16> = self.parameters.iter().map(|parameter| {
            return (parameter * NEURAL_QUANTIZATION_SCALE as f32).round() as i16;
        }).collect();
        let mut network = NeuralNetwork::create_zero_network();
        network.feature_weights.copy_from_slice(&weights[..FEATURE_PARAMETER_COUNT]);
        network.hidden_biases.copy_from_slice(&weights[FEATURE_PARAMETER_COUNT..OUTPUT_PARAMETER_OFFSET]);
        network.output_weights.copy_from_slice(&weights[OUTPUT_PARAMETER_OFFSET..OUTPUT_BIAS_OFFSET]);
        network.output_biases.copy_from_slice(&weights[OUTPUT_BIAS_OFFSET..]);
        network.has_trained_value = true;
        network.has_trained_policy = has_trained_policy;
        network.has_trained_outcome = has_trained_outcome;

        return network;
    }
}

fn compute_validation_loss(
    network: &FloatingNetwork,
    records: &[TrainingRecord],
) -> f32 {
    let mut loss = 0.0;
    let mut record_count = 0;

    for record in records.iter().filter(|record| record.is_validation) {
        loss += network.compute_loss_gradient(
            record,
            None,
        );
        record_count += 1;
    }

    return loss / record_count as f32;
}

pub(crate) struct TrainingDataset {
    pub(crate) records: Vec<TrainingRecord>,
    pub(crate) duplicate_record_count: usize,
    pub(crate) dataset_fingerprint: u64,
    pub(crate) training_record_count: usize,
    pub(crate) validation_record_count: usize,
}

pub(crate) fn load_training_dataset(
    dataset_path: &Path,
    maximum_record_count: usize,
) -> EngineResult<TrainingDataset> {
    let mut input = BufReader::new(File::open(dataset_path)?);
    let mut records = Vec::new();
    let mut seen_positions = HashSet::new();
    let mut duplicate_record_count = 0;
    let mut line_number = 0;
    let mut dataset_fingerprint = FINGERPRINT_OFFSET;

    loop {
        let mut bytes = Vec::new();
        let byte_count = input.by_ref().take(MAXIMUM_RECORD_BYTES as u64 + 1).read_until(
            b'\n',
            &mut bytes,
        )?;

        if byte_count == 0 {
            break;
        }

        line_number += 1;

        if byte_count > MAXIMUM_RECORD_BYTES {
            return Err(EngineError::create_invalid_input(&format!("Failed to read training record {line_number} because it exceeds the size limit.")));
        }

        update_byte_fingerprint(
            &mut dataset_fingerprint,
            &bytes,
        );

        let text = String::from_utf8(bytes).map_err(|_| EngineError::create_invalid_input("Failed to decode a training record."))?;

        if text.trim().is_empty() {
            continue;
        }

        match parse_training_record(
            text.trim(),
            &mut seen_positions,
        ).map_err(|error| EngineError::create_invalid_input(&format!("Failed to read training record {line_number}: {error}")))? {
            Some(record) => records.push(record),
            None => duplicate_record_count += 1,
        }

        if records.len() > maximum_record_count {
            return Err(EngineError::create_invalid_input("Failed to train the network because the dataset exceeds the configured record budget."));
        }
    }

    let validation_record_count = records.iter().filter(|record| record.is_validation).count();
    let training_record_count = records.len() - validation_record_count;

    if (training_record_count == 0) || (validation_record_count == 0) {
        return Err(EngineError::create_invalid_input("Failed to train the network because the game-based split requires both training and validation records."));
    }

    return Ok(TrainingDataset {
        records,
        duplicate_record_count,
        dataset_fingerprint,
        training_record_count,
        validation_record_count,
    });
}

pub fn train_network(
    dataset_path: &Path,
    output_path: &Path,
    options: &TrainingOptions,
    observer: &mut dyn FnMut(&str),
) -> EngineResult<TrainingReport> {
    let manifest_path = output_path.with_extension("training.json");

    if output_path.exists() || manifest_path.exists() || !(1..=10000).contains(&options.epoch_count)
        || !(1..=65536).contains(&options.batch_size) || !options.learning_rate.is_finite()
        || !(0.0..=0.1).contains(&options.learning_rate) || (options.learning_rate == 0.0)
        || !(2..=1000000).contains(&options.maximum_record_count) {
        return Err(EngineError::create_invalid_input("Failed to train the network because its output exists or its options are invalid."));
    }

    let dataset = load_training_dataset(
        dataset_path,
        options.maximum_record_count,
    )?;
    let records = dataset.records;
    let duplicate_record_count = dataset.duplicate_record_count;
    let dataset_fingerprint = dataset.dataset_fingerprint;
    let training_record_count = dataset.training_record_count;
    let validation_record_count = dataset.validation_record_count;

    let has_trained_policy = records.iter().any(|record| !record.is_validation && !record.policies.is_empty());
    let has_trained_outcome = records.iter().any(|record| !record.is_validation && record.outcome_target.is_some());
    let mut random = RandomSequence::create_sequence(options.seed);
    let mut network = FloatingNetwork::create_network(&mut random);
    let mut best_network = network.clone();
    let initial_validation_loss = compute_validation_loss(
        &network,
        &records,
    );
    let mut best_validation_loss = f32::INFINITY;
    let mut best_epoch = 0;
    let mut first_moments = vec![0.0f32; PARAMETER_COUNT];
    let mut second_moments = vec![0.0f32; PARAMETER_COUNT];
    let mut gradients = vec![0.0f32; PARAMETER_COUNT];
    let mut indices: Vec<usize> = records.iter().enumerate().filter_map(|(index, record)| {
        return (!record.is_validation).then_some(index);
    }).collect();
    let mut step_count = 0u64;

    observer(&format!("Loaded {training_record_count} training records and {validation_record_count} validation records; removed {duplicate_record_count} duplicates."));

    for epoch in 0..options.epoch_count {
        for index in (1..indices.len()).rev() {
            let shuffled_index = (random.generate_integer() % (index as u64 + 1)) as usize;
            indices.swap(
                index,
                shuffled_index,
            );
        }

        let mut training_loss = 0.0;

        for batch in indices.chunks(options.batch_size) {
            gradients.fill(0.0);

            for &index in batch {
                training_loss += network.compute_loss_gradient(
                    &records[index],
                    Some(&mut gradients),
                );
            }

            step_count += 1;
            let first_correction = 1.0 - ADAM_FIRST_DECAY.powf(step_count as f32);
            let second_correction = 1.0 - ADAM_SECOND_DECAY.powf(step_count as f32);

            for parameter_index in 0..PARAMETER_COUNT {
                let gradient = (gradients[parameter_index] / batch.len() as f32).clamp(
                    -1.0,
                    1.0,
                );
                first_moments[parameter_index] = ADAM_FIRST_DECAY * first_moments[parameter_index]
                    + (1.0 - ADAM_FIRST_DECAY) * gradient;
                second_moments[parameter_index] = ADAM_SECOND_DECAY * second_moments[parameter_index]
                    + (1.0 - ADAM_SECOND_DECAY) * gradient * gradient;
                let corrected_first = first_moments[parameter_index] / first_correction;
                let corrected_second = second_moments[parameter_index] / second_correction;
                let parameter = &mut network.parameters[parameter_index];
                *parameter -= options.learning_rate * (corrected_first / (corrected_second.sqrt() + ADAM_EPSILON) + WEIGHT_DECAY * *parameter);

                let bound = if parameter_index < OUTPUT_PARAMETER_OFFSET {
                    2.0
                } else {
                    8.0
                };
                *parameter = parameter.clamp(
                    -bound,
                    bound,
                );

                if !parameter.is_finite() {
                    return Err(EngineError::create_invalid_input("Failed to train the network because an optimizer update is non-finite."));
                }
            }
        }

        let validation_loss = compute_validation_loss(
            &network,
            &records,
        );

        if validation_loss < best_validation_loss {
            best_validation_loss = validation_loss;
            best_network = network.clone();
            best_epoch = epoch + 1;
        }

        observer(&format!(
            "Completed epoch {}: training loss {:.6}, validation loss {validation_loss:.6}.",
            epoch + 1,
            training_loss / training_record_count as f32,
        ));
    }

    let quantized = best_network.quantize_network(
        has_trained_policy,
        has_trained_outcome,
    );
    let mut maximum_quantization_error_centipawns = 0.0f32;

    for record in records.iter().filter(|record| record.is_validation).take(256) {
        let features = encode_features(&record.board);
        let mut accumulator = crate::neural::NeuralAccumulator {
            board: record.board,
            features,
            ..crate::neural::NeuralAccumulator::default()
        };

        for perspective in 0..2 {
            accumulator.sums[perspective] = quantized.hidden_biases.map(i32::from);

            for feature in &features.features[perspective][..features.counts[perspective]] {
                for channel in 0..NEURAL_HIDDEN_COUNT {
                    accumulator.sums[perspective][channel] += quantized.feature_weights[feature.index as usize * NEURAL_HIDDEN_COUNT + channel] as i32 * feature.count as i32;
                }
            }
        }

        let float_score = best_network.compute_forward_pass(record).outputs[0] * NEURAL_SCORE_SCALE as f32;
        let integer_score = quantized.evaluate_head(
            &accumulator,
            record.side,
            0,
        ) as f32 * NEURAL_SCORE_SCALE as f32;
        maximum_quantization_error_centipawns = maximum_quantization_error_centipawns.max((float_score - integer_score).abs());
    }

    quantized.save_network(output_path)?;
    let fingerprint = quantized.get_fingerprint();
    let manifest = format!(
        concat!(
            "{{\n",
            "    \"architecture_version\": 2,\n",
            "    \"training_backend\": \"rust cpu\",\n",
            "    \"optimizer\": \"adamw\",\n",
            "    \"dataset_fnv_fingerprint\": {dataset_fingerprint},\n",
            "    \"network_fingerprint\": {fingerprint},\n",
            "    \"seed\": {},\n",
            "    \"epoch_count\": {},\n",
            "    \"best_epoch\": {best_epoch},\n",
            "    \"batch_size\": {},\n",
            "    \"learning_rate\": {},\n",
            "    \"training_record_count\": {training_record_count},\n",
            "    \"validation_record_count\": {validation_record_count},\n",
            "    \"duplicate_record_count\": {duplicate_record_count},\n",
            "    \"initial_validation_loss\": {initial_validation_loss},\n",
            "    \"best_validation_loss\": {best_validation_loss},\n",
            "    \"maximum_sampled_quantization_error_centipawns\": {maximum_quantization_error_centipawns},\n",
            "    \"has_trained_policy\": {has_trained_policy},\n",
            "    \"has_trained_outcome\": {has_trained_outcome}\n",
            "}}\n",
        ),
        options.seed,
        options.epoch_count,
        options.batch_size,
        options.learning_rate,
        dataset_fingerprint = dataset_fingerprint,
        fingerprint = fingerprint,
        best_epoch = best_epoch,
        training_record_count = training_record_count,
        validation_record_count = validation_record_count,
        duplicate_record_count = duplicate_record_count,
        initial_validation_loss = initial_validation_loss,
        best_validation_loss = best_validation_loss,
        maximum_quantization_error_centipawns = maximum_quantization_error_centipawns,
        has_trained_policy = has_trained_policy,
        has_trained_outcome = has_trained_outcome,
    );
    let mut manifest_output = OpenOptions::new().write(true).create_new(true).open(manifest_path)?;
    manifest_output.write_all(manifest.as_bytes())?;
    manifest_output.flush()?;

    return Ok(TrainingReport {
        training_record_count,
        validation_record_count,
        duplicate_record_count,
        initial_validation_loss,
        best_validation_loss,
        maximum_quantization_error_centipawns,
        network_fingerprint: fingerprint,
        dataset_fingerprint,
    });
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::neural::{
        encode_features,
        NEURAL_HIDDEN_COUNT,
    };
    use super::{
        compute_cross_entropy,
        compute_sigmoid,
        FEATURE_PARAMETER_COUNT,
        FloatingNetwork,
        OUTPUT_BIAS_OFFSET,
        PARAMETER_COUNT,
        parse_training_record,
        RandomSequence,
    };

    #[test]
    fn test_value_outcome_and_policy_gradients() {
        let record = parse_training_record(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 100 | 1 | e2e4:0.7 d2d4:0.3 | gradient_game",
            &mut HashSet::new(),
        ).unwrap().unwrap();
        let mut network = FloatingNetwork::create_network(&mut RandomSequence::create_sequence(42));
        let mut gradients = vec![0.0; PARAMETER_COUNT];
        network.compute_loss_gradient(
            &record,
            Some(&mut gradients),
        );

        let feature_index = encode_features(&record.board).features[0][0].index as usize;
        let policy_head = record.policies.iter().find(|policy| policy.probability > 0.0).unwrap().heads[0];

        for parameter in [OUTPUT_BIAS_OFFSET, OUTPUT_BIAS_OFFSET + 1, OUTPUT_BIAS_OFFSET + policy_head, feature_index * NEURAL_HIDDEN_COUNT, FEATURE_PARAMETER_COUNT,] {
            let original = network.parameters[parameter];
            network.parameters[parameter] = original + 0.001;
            let upper_loss = network.compute_loss_gradient(
                &record,
                None,
            );
            network.parameters[parameter] = original - 0.001;
            let lower_loss = network.compute_loss_gradient(
                &record,
                None,
            );
            network.parameters[parameter] = original;
            let numerical_gradient = (upper_loss - lower_loss) / 0.002;
            assert!((numerical_gradient - gradients[parameter]).abs() < 0.002);
        }
    }

    #[test]
    fn test_training_score_and_outcome_perspectives() {
        let unknown = parse_training_record(
            "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1 | 400 | ? | ? | unknown_game",
            &mut HashSet::new(),
        ).unwrap().unwrap();
        assert!(unknown.outcome_target.is_none());
        assert!((unknown.value_target - 0.26894143).abs() < 0.000001);
        let black_win = parse_training_record(
            "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1 | -400 | 0 | ? | black_win",
            &mut HashSet::new(),
        ).unwrap().unwrap();
        assert_eq!(
            black_win.outcome_target,
            Some([1.0, 0.0, 0.0,]),
        );
        assert!(black_win.value_target > 0.731);
    }

    #[test]
    fn test_policy_targets_normalize_and_include_unlabeled_legal_moves() {
        let record = parse_training_record(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | ? | e2e4:7 d2d4:3 | policy_game",
            &mut HashSet::new(),
        ).unwrap().unwrap();
        assert_eq!(
            record.policies.len(),
            20,
        );
        assert!((record.policies.iter().map(|policy| policy.probability).sum::<f32>() - 1.0).abs() < 0.000001);
        let mut positive: Vec<f32> = record.policies.iter().map(|policy| policy.probability).filter(|&probability| probability > 0.0).collect();
        positive.sort_by(f32::total_cmp);
        assert_eq!(
            positive,
            vec![0.3, 0.7,],
        );
    }

    #[test]
    fn test_policy_parser_rejects_nonfinite_negative_duplicate_and_empty_labels() {
        for policy in ["e2e4:NaN", "e2e4:inf", "e2e4:-1", "e2e4:0", "e2e4:1 e2e4:2", "e2e4", "e2e4:3e38 d2d4:3e38", "",] {
            let record = format!("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | ? | {policy} | policy_game");
            assert!(parse_training_record(
                &record,
                &mut HashSet::new(),
            ).is_err());
        }
    }

    #[test]
    fn test_unknown_outcomes_do_not_generate_outcome_head_gradients() {
        let record = parse_training_record(
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1 | 100 | ? | ? | unknown_game",
            &mut HashSet::new(),
        ).unwrap().unwrap();
        let network = FloatingNetwork::create_network(&mut RandomSequence::create_sequence(42));
        let mut gradients = vec![0.0; PARAMETER_COUNT];
        network.compute_loss_gradient(
            &record,
            Some(&mut gradients),
        );
        assert_ne!(
            gradients[OUTPUT_BIAS_OFFSET],
            0.0,
        );
        assert!(gradients[(OUTPUT_BIAS_OFFSET + 1)..].iter().all(|&gradient| gradient == 0.0));
    }

    #[test]
    fn test_position_deduplication_ignores_clocks_and_game_identifiers() {
        let mut seen_positions = HashSet::new();
        let first = parse_training_record(
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1 | 0 | ? | ? | game_0",
            &mut seen_positions,
        ).unwrap();
        let duplicate = parse_training_record(
            "4k3/8/8/8/8/8/4P3/4K3 w - - 40 22 | 100 | 1 | ? | game_3",
            &mut seen_positions,
        ).unwrap();
        assert!(first.is_some());
        assert!(duplicate.is_none());
        assert_eq!(
            seen_positions.len(),
            1,
        );
    }

    #[test]
    fn test_game_split_is_stable_across_different_positions() {
        for game_id in ["game_0", "game_3",] {
            let first = parse_training_record(
                &format!("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1 | 0 | ? | ? | {game_id}"),
                &mut HashSet::new(),
            ).unwrap().unwrap();
            let second = parse_training_record(
                &format!("4k3/8/8/8/8/4P3/8/4K3 b - - 0 1 | 0 | ? | ? | {game_id}"),
                &mut HashSet::new(),
            ).unwrap().unwrap();
            assert_eq!(
                first.is_validation,
                second.is_validation,
            );
            assert_eq!(
                first.is_validation,
                game_id == "game_3",
            );
        }
    }

    #[test]
    fn test_loss_functions_remain_finite_at_extreme_scores() {
        for score in [-20000.0, -400.0, 0.0, 400.0, 20000.0,] {
            let prediction = compute_sigmoid(score);
            assert!(prediction.is_finite());
            assert!((0.0..=1.0).contains(&prediction));

            for target in [0.0, 0.5, 1.0,] {
                let loss = compute_cross_entropy(
                    prediction,
                    target,
                );
                assert!(loss.is_finite());
                assert!(loss >= 0.0);
            }
        }
    }
}
