use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{
    AtomicU64,
    Ordering,
};

use alekhine::network::Network;
use alekhine::neural::NeuralNetwork;
use alekhine::position::Position;
use alekhine::training::{
    train_network,
    TrainingOptions,
};
use alekhine::types::START_FEN;

const TRAINING_RECORDS: &str = concat!(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 100 | 1 | e2e4:0.7 d2d4:0.3 | game_0\n",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 | 100 | 1 | e7e5:1 | game_0\n",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1 | 100 | 1 | d7d5:1 | game_3\n",
);
static DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TrainingFixture {
    directory: PathBuf,
    dataset_path: PathBuf,
    output_path: PathBuf,
}

impl TrainingFixture {
    fn create_fixture() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "alekhine_training_tests_{}_{}",
            std::process::id(),
            DIRECTORY_SEQUENCE.fetch_add(
                1,
                Ordering::SeqCst,
            ),
        ));
        fs::create_dir(&directory).unwrap();
        let dataset_path = directory.join("positions.txt");
        fs::write(
            &dataset_path,
            TRAINING_RECORDS,
        ).unwrap();

        return Self {
            output_path: directory.join("model.nnue"),
            directory,
            dataset_path,
        };
    }
}

impl Drop for TrainingFixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).expect("Failed to remove the temporary training fixture.");
    }
}

fn create_training_options() -> TrainingOptions {
    return TrainingOptions {
        epoch_count: 2,
        batch_size: 2,
        seed: 42,
        ..TrainingOptions::default()
    };
}

#[test]
fn test_training_is_reproducible_and_exports_loadable_artifacts() {
    let fixture = TrainingFixture::create_fixture();
    let mut messages = Vec::new();
    let first = train_network(
        &fixture.dataset_path,
        &fixture.output_path,
        &create_training_options(),
        &mut |message| messages.push(message.to_owned()),
    ).unwrap();
    let second_path = fixture.directory.join("second.nnue");
    let second = train_network(
        &fixture.dataset_path,
        &second_path,
        &create_training_options(),
        &mut |_| (),
    ).unwrap();
    assert_eq!(
        fs::read(&fixture.output_path).unwrap(),
        fs::read(second_path).unwrap(),
    );
    assert_eq!(
        first.network_fingerprint,
        second.network_fingerprint,
    );
    assert_eq!(
        (first.training_record_count, first.validation_record_count,),
        (2, 1,),
    );
    assert!(first.initial_validation_loss.is_finite());
    assert!(first.best_validation_loss.is_finite());
    assert!(first.maximum_quantization_error_centipawns.is_finite());
    assert!(messages.iter().any(|message| message.starts_with("Completed epoch 2:")));
    let network = Network::load_network(&fixture.output_path).unwrap();
    assert_eq!(
        network.get_fingerprint(),
        first.network_fingerprint,
    );
    let position = Position::parse_fen(START_FEN).unwrap();
    let score = network.evaluate_position(
        &position,
        &network.build_accumulator(&position),
    );
    assert!(score.abs() < alekhine::types::MATE_THRESHOLD);
    let manifest = fs::read_to_string(fixture.output_path.with_extension("training.json")).unwrap();
    assert!(manifest.contains(&format!("\"network_fingerprint\": {}", first.network_fingerprint)));
    assert!(manifest.contains(&format!("\"dataset_fnv_fingerprint\": {}", first.dataset_fingerprint)));
}

#[test]
fn test_training_flags_only_reflect_labels_in_training_partition() {
    let fixture = TrainingFixture::create_fixture();
    let records = TRAINING_RECORDS.lines().enumerate().map(|(index, line)| {
        if index < 2 {
            let fields: Vec<&str> = line.split('|').collect();

            return format!("{} | {} | ? | ? | game_0", fields[0].trim(), fields[1].trim());
        }

        return line.to_owned();
    }).collect::<Vec<String>>().join("\n") + "\n";
    fs::write(
        &fixture.dataset_path,
        records,
    ).unwrap();
    train_network(
        &fixture.dataset_path,
        &fixture.output_path,
        &create_training_options(),
        &mut |_| (),
    ).unwrap();
    let network = NeuralNetwork::load_network(&fs::read(&fixture.output_path).unwrap()).unwrap();
    assert!(network.has_trained_value);
    assert!(!network.has_trained_policy);
    assert!(!network.has_trained_outcome);
}

#[test]
fn test_training_deduplicates_before_counting_partitions() {
    let fixture = TrainingFixture::create_fixture();
    fs::write(
        &fixture.dataset_path,
        format!("{TRAINING_RECORDS}\n{START_FEN} | 200 | 0 | ? | game_3\n"),
    ).unwrap();
    let report = train_network(
        &fixture.dataset_path,
        &fixture.output_path,
        &create_training_options(),
        &mut |_| (),
    ).unwrap();
    assert_eq!(
        (report.training_record_count, report.validation_record_count, report.duplicate_record_count,),
        (2, 1, 1,),
    );
}

#[test]
fn test_training_rejects_invalid_options_without_creating_outputs() {
    let fixture = TrainingFixture::create_fixture();

    for options in [
        TrainingOptions {
            epoch_count: 0,
            ..create_training_options()
        },
        TrainingOptions {
            epoch_count: 10001,
            ..create_training_options()
        },
        TrainingOptions {
            batch_size: 0,
            ..create_training_options()
        },
        TrainingOptions {
            batch_size: 65537,
            ..create_training_options()
        },
        TrainingOptions {
            learning_rate: f32::NAN,
            ..create_training_options()
        },
        TrainingOptions {
            learning_rate: f32::INFINITY,
            ..create_training_options()
        },
        TrainingOptions {
            learning_rate: -0.1,
            ..create_training_options()
        },
        TrainingOptions {
            learning_rate: 0.0,
            ..create_training_options()
        },
        TrainingOptions {
            learning_rate: 0.11,
            ..create_training_options()
        },
        TrainingOptions {
            maximum_record_count: 1,
            ..create_training_options()
        },
        TrainingOptions {
            maximum_record_count: 1000001,
            ..create_training_options()
        },
    ] {
        assert!(train_network(
            &fixture.dataset_path,
            &fixture.output_path,
            &options,
            &mut |_| (),
        ).is_err());
        assert!(!fixture.output_path.exists());
        assert!(!fixture.output_path.with_extension("training.json").exists());
    }
}

#[test]
fn test_training_rejects_record_budget_overflow_before_export() {
    let fixture = TrainingFixture::create_fixture();
    let error = train_network(
        &fixture.dataset_path,
        &fixture.output_path,
        &TrainingOptions {
            maximum_record_count: 2,
            ..create_training_options()
        },
        &mut |_| (),
    ).err().unwrap();
    assert!(error.to_string().contains("record budget"));
    assert!(!fixture.output_path.exists());
}

#[test]
fn test_training_rejects_empty_oversized_and_non_utf8_datasets() {
    let fixture = TrainingFixture::create_fixture();

    for bytes in [Vec::new(), vec![b'x'; 8193], vec![0xff, b'\n',],] {
        fs::write(
            &fixture.dataset_path,
            bytes,
        ).unwrap();
        assert!(train_network(
            &fixture.dataset_path,
            &fixture.output_path,
            &create_training_options(),
            &mut |_| (),
        ).is_err());
        assert!(!fixture.output_path.exists());
    }
}

#[test]
fn test_training_reports_record_number_and_preserves_existing_outputs() {
    let fixture = TrainingFixture::create_fixture();
    fs::write(
        &fixture.dataset_path,
        format!("{TRAINING_RECORDS}Invalid record.\n"),
    ).unwrap();
    let error = train_network(
        &fixture.dataset_path,
        &fixture.output_path,
        &create_training_options(),
        &mut |_| (),
    ).err().unwrap();
    assert!(error.to_string().contains("record 4"));

    for path in [&fixture.output_path, &fixture.output_path.with_extension("training.json"),] {
        fs::write(
            path,
            b"Preserved output.\n",
        ).unwrap();
        assert!(train_network(
            &fixture.dataset_path,
            &fixture.output_path,
            &create_training_options(),
            &mut |_| (),
        ).is_err());
        assert_eq!(
            fs::read(path).unwrap(),
            b"Preserved output.\n",
        );
        fs::remove_file(path).unwrap();
    }
}
