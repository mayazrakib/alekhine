use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use crate::neural::NEURAL_SCORE_SCALE;
use crate::stockfish_network::{
    DENSE_BIAS_SCALE,
    DENSE_FIRST_OUTPUT_COUNT,
    DENSE_FORWARD_SCALE,
    DENSE_INPUT_STRIDE,
    DENSE_OUTPUT_SCALE,
    DENSE_SECOND_INPUT_COUNT,
    DENSE_WEIGHT_SCALE,
    DenseStack,
    STOCKFISH_BUCKET_COUNT,
    StockfishFeatures,
    StockfishNetwork,
};
use crate::training::{
    ADAM_EPSILON,
    ADAM_FIRST_DECAY,
    ADAM_SECOND_DECAY,
    compute_cross_entropy,
    compute_sigmoid,
    load_training_dataset,
    RandomSequence,
    TrainingOptions,
    WEIGHT_DECAY,
};
use crate::types::{
    EngineError,
    EngineResult,
};

const MAXIMUM_BIAS_MAGNITUDE: f64 = 100_000_000.0 / DENSE_BIAS_SCALE;
const SQUARED_ACTIVATION_FACTOR: f64 = 127.0 / 128.0;

pub struct StockfishTrainingReport {
    pub source_fingerprint: u64,
    pub network_fingerprint: u64,
    pub best_epoch: usize,
    pub initial_validation_loss: f64,
    pub best_validation_loss: f64,
    pub changed_parameter_count: usize,
    pub trainable_parameter_count: usize,
    pub maximum_quantization_error_centipawns: f64,
}

#[derive(Clone)]
struct FloatingLayer {
    biases: Vec<f64>,
    weights: Vec<f64>,
    input_count: usize,
}

#[derive(Clone)]
struct FloatingStack {
    layers: [FloatingLayer; 3],
}

struct CachedRecord {
    features: StockfishFeatures,
    value_target: f32,
    is_validation: bool,
}

struct ForwardPass {
    inputs: Vec<f64>,
    first_outputs: Vec<f64>,
    second_inputs: Vec<f64>,
    second_outputs: Vec<f64>,
    final_inputs: Vec<f64>,
    centipawn_score: f64,
}

fn import_dense_stack(stack: &DenseStack) -> FloatingStack {
    return FloatingStack {
        layers: std::array::from_fn(|index| FloatingLayer {
            biases: stack.layers[index].biases.iter().map(|&bias| f64::from(bias) / DENSE_BIAS_SCALE).collect(),
            weights: stack.layers[index].weights.iter().map(|&weight| f64::from(weight) / DENSE_WEIGHT_SCALE).collect(),
            input_count: stack.layers[index].input_count,
        }),
    };
}

fn create_zero_stack(stack: &FloatingStack) -> FloatingStack {
    return FloatingStack {
        layers: std::array::from_fn(|index| FloatingLayer {
            biases: vec![0.0; stack.layers[index].biases.len()],
            weights: vec![0.0; stack.layers[index].weights.len()],
            input_count: stack.layers[index].input_count,
        }),
    };
}

fn calculate_layer_outputs(
    layer: &FloatingLayer,
    inputs: &[f64],
) -> Vec<f64> {
    return layer.biases.iter().enumerate().map(|(output_index, &bias)| {
        let weights = &layer.weights[output_index * layer.input_count..(output_index + 1) * layer.input_count];

        return weights.iter().zip(inputs).fold(
            bias,
            |sum, (&weight, &input)| sum + weight * input,
        );
    }).collect();
}

fn calculate_forward_pass(
    stack: &FloatingStack,
    features: &StockfishFeatures,
) -> ForwardPass {
    let inputs: Vec<f64> = features.inputs.iter().map(|&activation| f64::from(activation) / 127.0).collect();
    let first_outputs = calculate_layer_outputs(
        &stack.layers[0],
        &inputs,
    );
    let mut second_inputs = vec![0.0; DENSE_INPUT_STRIDE];

    for channel in 0..DENSE_FIRST_OUTPUT_COUNT - 1 {
        second_inputs[channel] = (first_outputs[channel].powi(2) * SQUARED_ACTIVATION_FACTOR).min(1.0);
        second_inputs[channel + DENSE_FIRST_OUTPUT_COUNT - 1] = first_outputs[channel].clamp(
            0.0,
            1.0,
        );
    }

    let second_outputs = calculate_layer_outputs(
        &stack.layers[1],
        &second_inputs,
    );
    let final_inputs: Vec<f64> = second_outputs.iter().map(|&activation| activation.clamp(
        0.0,
        1.0,
    )).collect();
    let final_output = calculate_layer_outputs(
        &stack.layers[2],
        &final_inputs,
    )[0];
    let native_score = f64::from(features.psqt_scores[features.bucket_index])
        + DENSE_OUTPUT_SCALE * final_output
        + DENSE_FORWARD_SCALE * first_outputs[DENSE_FIRST_OUTPUT_COUNT - 1];

    return ForwardPass {
        inputs,
        first_outputs,
        second_inputs,
        second_outputs,
        final_inputs,
        centipawn_score: native_score * features.centipawn_scale,
    };
}

fn add_layer_gradients(
    layer: &FloatingLayer,
    inputs: &[f64],
    output_gradients: &[f64],
    gradients: &mut FloatingLayer,
) -> Vec<f64> {
    let mut input_gradients = vec![0.0; layer.input_count];

    for (output_index, &gradient,) in output_gradients.iter().enumerate() {
        gradients.biases[output_index] += gradient;

        for (input_index, &input,) in inputs.iter().enumerate() {
            let parameter_index = output_index * layer.input_count + input_index;
            gradients.weights[parameter_index] += gradient * input;
            input_gradients[input_index] += gradient * layer.weights[parameter_index];
        }
    }

    return input_gradients;
}

fn compute_loss_gradient(
    stack: &FloatingStack,
    record: &CachedRecord,
    gradients: Option<&mut FloatingStack>,
) -> f64 {
    let forward = calculate_forward_pass(
        stack,
        &record.features,
    );
    let probability = compute_sigmoid((forward.centipawn_score / f64::from(NEURAL_SCORE_SCALE)) as f32);
    let loss = f64::from(compute_cross_entropy(
        probability,
        record.value_target,
    ));
    let Some(gradients) = gradients else {
        return loss;
    };
    let native_gradient = f64::from(probability - record.value_target)
        * record.features.centipawn_scale / f64::from(NEURAL_SCORE_SCALE);
    let final_gradients = add_layer_gradients(
        &stack.layers[2],
        &forward.final_inputs,
        &[native_gradient * DENSE_OUTPUT_SCALE],
        &mut gradients.layers[2],
    );
    let second_gradients: Vec<f64> = final_gradients.iter().zip(&forward.second_outputs).map(|(&gradient, &activation)| {
        return if (activation > 0.0) && (activation < 1.0) { gradient } else { 0.0 };
    }).collect();
    let activation_gradients = add_layer_gradients(
        &stack.layers[1],
        &forward.second_inputs,
        &second_gradients,
        &mut gradients.layers[1],
    );
    let mut first_gradients = vec![0.0; DENSE_FIRST_OUTPUT_COUNT];

    for channel in 0..DENSE_FIRST_OUTPUT_COUNT - 1 {
        let activation = forward.first_outputs[channel];

        if activation.powi(2) * SQUARED_ACTIVATION_FACTOR < 1.0 {
            first_gradients[channel] += activation_gradients[channel] * 2.0 * activation * SQUARED_ACTIVATION_FACTOR;
        }

        if (activation > 0.0) && (activation < 1.0) {
            first_gradients[channel] += activation_gradients[channel + DENSE_FIRST_OUTPUT_COUNT - 1];
        }
    }

    first_gradients[DENSE_FIRST_OUTPUT_COUNT - 1] = native_gradient * DENSE_FORWARD_SCALE;
    add_layer_gradients(
        &stack.layers[0],
        &forward.inputs,
        &first_gradients,
        &mut gradients.layers[0],
    );

    return loss;
}

fn quantize_stacks(
    source: &StockfishNetwork,
    stacks: &[FloatingStack; STOCKFISH_BUCKET_COUNT],
) -> StockfishNetwork {
    let mut network = source.clone();

    for (bucket_index, stack,) in stacks.iter().enumerate() {
        for (layer_index, layer,) in stack.layers.iter().enumerate() {
            for (parameter, &bias,) in network.stacks[bucket_index].layers[layer_index].biases.iter_mut().zip(&layer.biases) {
                *parameter = (bias * DENSE_BIAS_SCALE).round() as i32;
            }

            for (parameter, &weight,) in network.stacks[bucket_index].layers[layer_index].weights.iter_mut().zip(&layer.weights) {
                *parameter = (weight * DENSE_WEIGHT_SCALE).round().clamp(
                    -128.0,
                    127.0,
                ) as i8;
            }
        }
    }

    return network;
}

fn compute_validation_loss(
    network: &StockfishNetwork,
    records: &[CachedRecord],
) -> f64 {
    let mut total_loss = 0.0;
    let mut record_count = 0;

    for record in records.iter().filter(|record| record.is_validation) {
        let score = network.evaluate_features(
            &record.features,
            record.features.bucket_index,
        ).centipawn_score;
        total_loss += f64::from(compute_cross_entropy(
            compute_sigmoid(score as f32 / NEURAL_SCORE_SCALE as f32),
            record.value_target,
        ));
        record_count += 1;
    }

    return total_loss / record_count as f64;
}

fn update_parameter(
    parameter: &mut f64,
    gradient: f64,
    first_moment: &mut f64,
    second_moment: &mut f64,
    correction_factors: [f64; 2],
    options: &TrainingOptions,
) {
    let gradient = gradient.clamp(
        -1.0,
        1.0,
    );
    *first_moment = f64::from(ADAM_FIRST_DECAY) * *first_moment + (1.0 - f64::from(ADAM_FIRST_DECAY)) * gradient;
    *second_moment = f64::from(ADAM_SECOND_DECAY) * *second_moment + (1.0 - f64::from(ADAM_SECOND_DECAY)) * gradient.powi(2);
    let corrected_first = *first_moment / correction_factors[0];
    let corrected_second = *second_moment / correction_factors[1];
    *parameter -= f64::from(options.learning_rate)
        * (corrected_first / (corrected_second.sqrt() + f64::from(ADAM_EPSILON)) + f64::from(WEIGHT_DECAY) * *parameter);
}

fn update_stack(
    stack: &mut FloatingStack,
    gradients: &FloatingStack,
    first_moments: &mut FloatingStack,
    second_moments: &mut FloatingStack,
    batch_count: usize,
    step_count: u64,
    options: &TrainingOptions,
) -> EngineResult<()> {
    let correction_factors = [
        1.0 - f64::from(ADAM_FIRST_DECAY).powf(step_count as f64),
        1.0 - f64::from(ADAM_SECOND_DECAY).powf(step_count as f64),
    ];

    for (layer_index, layer,) in stack.layers.iter_mut().enumerate() {
        for (index, parameter,) in layer.biases.iter_mut().enumerate() {
            update_parameter(
                parameter,
                gradients.layers[layer_index].biases[index] / batch_count as f64,
                &mut first_moments.layers[layer_index].biases[index],
                &mut second_moments.layers[layer_index].biases[index],
                correction_factors,
                options,
            );
            *parameter = parameter.clamp(
                -MAXIMUM_BIAS_MAGNITUDE,
                MAXIMUM_BIAS_MAGNITUDE,
            );
        }

        for (index, parameter,) in layer.weights.iter_mut().enumerate() {
            if (layer_index == 1) && (index % layer.input_count >= DENSE_SECOND_INPUT_COUNT) {
                continue;
            }

            update_parameter(
                parameter,
                gradients.layers[layer_index].weights[index] / batch_count as f64,
                &mut first_moments.layers[layer_index].weights[index],
                &mut second_moments.layers[layer_index].weights[index],
                correction_factors,
                options,
            );
            *parameter = parameter.clamp(
                -128.0 / DENSE_WEIGHT_SCALE,
                127.0 / DENSE_WEIGHT_SCALE,
            );
        }

        if layer.biases.iter().chain(&layer.weights).any(|parameter| !parameter.is_finite()) {
            return Err(EngineError::create_invalid_input("Failed to fine-tune Stockfish weights because an optimizer update is nonfinite."));
        }
    }

    return Ok(());
}

pub fn fine_tune_stockfish(
    source_path: &Path,
    dataset_path: &Path,
    output_path: &Path,
    options: &TrainingOptions,
    observer: &mut dyn FnMut(&str),
) -> EngineResult<StockfishTrainingReport> {
    let manifest_path = output_path.with_extension("training.json");

    if output_path.exists() || manifest_path.exists() || !(1..=10000).contains(&options.epoch_count)
        || !(1..=65536).contains(&options.batch_size) || !options.learning_rate.is_finite()
        || !(0.0..=0.1).contains(&options.learning_rate) || (options.learning_rate == 0.0)
        || !(2..=1000000).contains(&options.maximum_record_count) {
        return Err(EngineError::create_invalid_input("Failed to fine-tune Stockfish weights because the output exists or the options are invalid."));
    }

    let source = StockfishNetwork::load_network(source_path)?;
    let source_fingerprint = source.get_fingerprint();
    let dataset = load_training_dataset(
        dataset_path,
        options.maximum_record_count,
    )?;
    let mut records = Vec::with_capacity(dataset.records.len());
    let mut has_training_records = [false; STOCKFISH_BUCKET_COUNT];

    for record in dataset.records {
        let features = source.build_features(
            &record.board,
            record.side,
        );
        has_training_records[features.bucket_index] |= !record.is_validation;
        records.push(CachedRecord {
            features,
            value_target: record.value_target,
            is_validation: record.is_validation,
        });
    }

    let mut stacks = std::array::from_fn(|index| import_dense_stack(&source.stacks[index]));
    let mut best_stacks = stacks.clone();
    let mut first_moments: [FloatingStack; STOCKFISH_BUCKET_COUNT] = std::array::from_fn(|index| create_zero_stack(&stacks[index]));
    let mut second_moments: [FloatingStack; STOCKFISH_BUCKET_COUNT] = std::array::from_fn(|index| create_zero_stack(&stacks[index]));
    let mut gradients: [FloatingStack; STOCKFISH_BUCKET_COUNT] = std::array::from_fn(|index| create_zero_stack(&stacks[index]));
    let initial_validation_loss = compute_validation_loss(
        &source,
        &records,
    );
    let mut best_validation_loss = initial_validation_loss;
    let mut best_epoch = 0;
    let mut best_network = source.clone();
    let mut indices: Vec<usize> = records.iter().enumerate().filter_map(|(index, record)| (!record.is_validation).then_some(index)).collect();
    let mut random = RandomSequence::create_sequence(options.seed);
    let mut step_count = 0;
    observer(&format!(
        "Loaded Stockfish 18 weights with {} hidden channels, {} training records, and {} validation records. The feature transformer is frozen.",
        source.get_hidden_width(),
        dataset.training_record_count,
        dataset.validation_record_count,
    ));

    for epoch in 0..options.epoch_count {
        for index in (1..indices.len()).rev() {
            indices.swap(
                index,
                random.generate_integer() as usize % (index + 1),
            );
        }

        let mut training_loss = 0.0;

        for batch in indices.chunks(options.batch_size) {
            for stack in &mut gradients {
                for layer in &mut stack.layers {
                    layer.biases.fill(0.0);
                    layer.weights.fill(0.0);
                }
            }

            for &index in batch {
                let record = &records[index];
                let bucket = record.features.bucket_index;
                training_loss += compute_loss_gradient(
                    &stacks[bucket],
                    record,
                    Some(&mut gradients[bucket]),
                );
            }

            step_count += 1;

            for bucket in 0..STOCKFISH_BUCKET_COUNT {
                if has_training_records[bucket] {
                    update_stack(
                        &mut stacks[bucket],
                        &gradients[bucket],
                        &mut first_moments[bucket],
                        &mut second_moments[bucket],
                        batch.len(),
                        step_count,
                        options,
                    )?;
                }
            }
        }

        let candidate = quantize_stacks(
            &source,
            &stacks,
        );
        let validation_loss = compute_validation_loss(
            &candidate,
            &records,
        );

        if validation_loss < best_validation_loss {
            best_validation_loss = validation_loss;
            best_epoch = epoch + 1;
            best_network = candidate;
            best_stacks = stacks.clone();
        }

        observer(&format!(
            "Completed fine-tuning epoch {}: training loss {:.6}, quantized validation loss {validation_loss:.6}.",
            epoch + 1,
            training_loss / dataset.training_record_count as f64,
        ));
    }

    let mut changed_parameter_count = 0;
    let mut trainable_parameter_count = 0;

    for (bucket, &has_training_records,) in has_training_records.iter().enumerate() {
        for layer_index in 0..3 {
            let original = &source.stacks[bucket].layers[layer_index];
            let selected = &best_network.stacks[bucket].layers[layer_index];
            changed_parameter_count += original.biases.iter().zip(&selected.biases).filter(|(original, selected)| original != selected).count();
            changed_parameter_count += original.weights.iter().zip(&selected.weights).filter(|(original, selected)| original != selected).count();

            if has_training_records {
                let input_count = if layer_index == 1 { DENSE_SECOND_INPUT_COUNT } else { original.input_count };
                trainable_parameter_count += original.biases.len() * (input_count + 1);
            }
        }
    }

    let mut maximum_quantization_error_centipawns = 0.0_f64;

    for record in records.iter().filter(|record| record.is_validation).take(256) {
        let floating_score = calculate_forward_pass(
            &best_stacks[record.features.bucket_index],
            &record.features,
        ).centipawn_score;
        let quantized_score = best_network.evaluate_features(
            &record.features,
            record.features.bucket_index,
        ).centipawn_score;
        maximum_quantization_error_centipawns = maximum_quantization_error_centipawns.max((floating_score - f64::from(quantized_score)).abs());
    }

    let network_fingerprint = best_network.get_fingerprint();
    best_network.save_network(output_path)?;
    let mut manifest = OpenOptions::new().write(true).create_new(true).open(manifest_path)?;
    let architecture = if source.get_hidden_width() == 1024 { "stockfish 18 big" } else { "stockfish 18 small" };
    writeln!(
        manifest,
        "{{\n    \"architecture\":\"{architecture}\",\n    \"training_backend\":\"rust cpu\",\n    \"optimizer\":\"adamw\",\n    \"source_fingerprint\":{source_fingerprint},\n    \"network_fingerprint\":{network_fingerprint},\n    \"dataset_fingerprint\":{},\n    \"is_feature_transformer_frozen\":true,\n    \"are_psqt_weights_frozen\":true,\n    \"are_policy_targets_used\":false,\n    \"training_record_count\":{},\n    \"validation_record_count\":{},\n    \"duplicate_record_count\":{},\n    \"epoch_count\":{},\n    \"best_epoch\":{best_epoch},\n    \"batch_size\":{},\n    \"learning_rate\":{},\n    \"seed\":{},\n    \"trainable_parameter_count\":{trainable_parameter_count},\n    \"changed_parameter_count\":{changed_parameter_count},\n    \"initial_validation_loss\":{initial_validation_loss},\n    \"best_validation_loss\":{best_validation_loss},\n    \"maximum_sampled_quantization_error_centipawns\":{maximum_quantization_error_centipawns}\n}}",
        dataset.dataset_fingerprint,
        dataset.training_record_count,
        dataset.validation_record_count,
        dataset.duplicate_record_count,
        options.epoch_count,
        options.batch_size,
        options.learning_rate,
        options.seed,
    )?;

    return Ok(StockfishTrainingReport {
        source_fingerprint,
        network_fingerprint,
        best_epoch,
        initial_validation_loss,
        best_validation_loss,
        changed_parameter_count,
        trainable_parameter_count,
        maximum_quantization_error_centipawns,
    });
}

#[cfg(test)]
mod tests {
    use super::{
        calculate_forward_pass,
        compute_loss_gradient,
        create_zero_stack,
        import_dense_stack,
        update_stack,
        CachedRecord,
        FloatingStack,
    };
    use crate::stockfish_network::{
        DenseLayer,
        DenseStack,
        STOCKFISH_BUCKET_COUNT,
        StockfishFeatures,
    };
    use crate::training::TrainingOptions;

    fn create_stack() -> FloatingStack {
        let dense_stack = DenseStack {
            layers: [
                DenseLayer {
                    biases: (0..16).map(|index| if index % 2 == 0 { -2000 } else { 3000 }).collect(),
                    weights: vec![3; 16 * 4],
                    input_count: 4,
                },
                DenseLayer {
                    biases: vec![1000; 32],
                    weights: vec![2; 32 * 32],
                    input_count: 32,
                },
                DenseLayer {
                    biases: vec![100],
                    weights: vec![5; 32],
                    input_count: 32,
                },
            ],
        };

        return import_dense_stack(&dense_stack);
    }

    fn create_record() -> CachedRecord {
        return CachedRecord {
            features: StockfishFeatures {
                inputs: vec![30, 50, 70, 90,],
                psqt_scores: [20; STOCKFISH_BUCKET_COUNT],
                bucket_index: 0,
                centipawn_scale: 0.26,
            },
            value_target: 0.8,
            is_validation: false,
        };
    }

    fn calculate_precise_loss(stack: &FloatingStack) -> f64 {
        let logit = calculate_forward_pass(
            stack,
            &create_record().features,
        ).centipawn_score / 400.0;

        return logit.exp().ln_1p() - f64::from(create_record().value_target) * logit;
    }

    #[test]
    fn test_dense_gradients_match_finite_differences() {
        let stack = create_stack();
        let mut gradients = create_zero_stack(&stack);
        compute_loss_gradient(
            &stack,
            &create_record(),
            Some(&mut gradients),
        );
        let epsilon = 0.00001;

        for layer in 0..3 {
            for is_bias in [true, false,] {
                let indices = if is_bias { vec![0, stack.layers[layer].biases.len() - 1,] } else { vec![0, stack.layers[layer].weights.len() - 1,] };

                for index in indices {
                    let mut positive = stack.clone();
                    let mut negative = stack.clone();
                    let expected = if is_bias {
                        positive.layers[layer].biases[index] += epsilon;
                        negative.layers[layer].biases[index] -= epsilon;
                        gradients.layers[layer].biases[index]
                    } else {
                        positive.layers[layer].weights[index] += epsilon;
                        negative.layers[layer].weights[index] -= epsilon;
                        gradients.layers[layer].weights[index]
                    };
                    let numerical = (calculate_precise_loss(&positive) - calculate_precise_loss(&negative)) / (2.0 * epsilon);
                    assert!(
                        (expected - numerical).abs() < 0.000001,
                        "The gradient differs in layer {layer}, parameter {index}: {expected} versus {numerical}.",
                    );
                }
            }
        }
    }

    #[test]
    fn test_optimizer_preserves_padding_and_updates_trainable_parameters() {
        let mut stack = create_stack();
        let padding = stack.layers[1].weights[30];
        let original_bias = stack.layers[0].biases[15];
        let mut gradients = create_zero_stack(&stack);
        compute_loss_gradient(
            &stack,
            &create_record(),
            Some(&mut gradients),
        );
        let mut first_moments = create_zero_stack(&stack);
        let mut second_moments = create_zero_stack(&stack);
        update_stack(
            &mut stack,
            &gradients,
            &mut first_moments,
            &mut second_moments,
            1,
            1,
            &TrainingOptions::default(),
        ).unwrap();
        assert_eq!(
            stack.layers[1].weights[30],
            padding,
        );
        assert_ne!(
            stack.layers[0].biases[15],
            original_bias,
        );
    }

    #[test]
    fn test_optimizer_rejects_nonfinite_gradients() {
        let mut stack = create_stack();
        let mut gradients = create_zero_stack(&stack);
        gradients.layers[0].biases[0] = f64::NAN;
        let mut first_moments = create_zero_stack(&stack);
        let mut second_moments = create_zero_stack(&stack);
        assert!(update_stack(
            &mut stack,
            &gradients,
            &mut first_moments,
            &mut second_moments,
            1,
            1,
            &TrainingOptions::default(),
        ).is_err());
    }
}
