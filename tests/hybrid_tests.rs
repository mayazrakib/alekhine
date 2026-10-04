use std::fs;
use std::sync::atomic::{
    AtomicU64,
    Ordering,
};

use alekhine::network::Network;
use alekhine::neural::{
    encode_features,
    NEURAL_HIDDEN_COUNT,
    NEURAL_QUANTIZATION_SCALE,
    NeuralNetwork,
};
use alekhine::position::Position;
use alekhine::search::{
    Search,
    SearchControl,
    SearchLimits,
    SearchOptions,
};
use alekhine::tactical::calculate_exchange_score;
use alekhine::training::{
    train_network,
    TrainingOptions,
};
use alekhine::transposition_table::TranspositionTable;
use alekhine::types::{
    Color,
    START_FEN,
};

static FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn create_test_network() -> NeuralNetwork {
    let mut network = NeuralNetwork::create_zero_network();

    for (index, weight) in network.feature_weights.iter_mut().enumerate() {
        *weight = ((index * 17 + 3) % 97) as i16 - 48;
    }

    for (index, weight) in network.output_weights.iter_mut().enumerate() {
        *weight = ((index * 19 + 7) % 31) as i16 - 15;
    }

    network.hidden_biases.fill(NEURAL_QUANTIZATION_SCALE as i16 / 2);
    network.has_trained_value = true;
    network.has_trained_policy = true;
    network.has_trained_outcome = true;

    return network;
}

fn get_temporary_path(name: &str) -> std::path::PathBuf {
    return std::env::temp_dir().join(format!(
        "alekhine_hybrid_{}_{}_{}",
        std::process::id(),
        FILE_SEQUENCE.fetch_add(
            1,
            Ordering::SeqCst,
        ),
        name,
    ));
}

#[test]
fn test_neural_accumulators_for_special_moves_and_king_buckets() {
    let network = Network::create_neural_network(create_test_network());

    for fen in [
        START_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "7k/8/8/8/3K4/8/8/R7 w - - 0 1",
        "4k3/8/8/8/3r4/3P4/3R4/4K3 w - - 0 1",
    ] {
        let mut position = Position::parse_fen(fen).unwrap();
        let original_fen = position.get_fen();
        let accumulator = network.build_accumulator(&position);
        let moves = position.generate_legal_moves(false).unwrap();

        for &chess_move in &moves.moves[..moves.count] {
            let undo = position.make_move(chess_move).unwrap();
            let incremental = network.update_accumulator(
                &accumulator,
                &undo,
            );
            assert_eq!(
                incremental,
                network.build_accumulator(&position),
            );
            position.unmake_move(&undo);
            assert_eq!(
                position.get_fen(),
                original_fen,
            );
        }
    }
}

#[test]
fn test_neural_accumulators_along_random_games() {
    let network = Network::create_neural_network(create_test_network());
    let mut seed = 20261004u64;

    for _ in 0..8 {
        let mut position = Position::parse_fen(START_FEN).unwrap();
        let mut accumulator = network.build_accumulator(&position);

        for _ in 0..128 {
            let moves = position.generate_legal_moves(false).unwrap();

            if moves.count == 0 {
                break;
            }

            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let chess_move = moves.moves[(seed % moves.count as u64) as usize];
            let undo = position.make_move(chess_move).unwrap();
            accumulator = network.update_accumulator(
                &accumulator,
                &undo,
            );
            assert_eq!(
                accumulator,
                network.build_accumulator(&position),
            );
        }
    }
}

#[test]
fn test_neural_color_symmetry_and_policy_orientation() {
    let network = create_test_network();
    let mut first = Position::parse_fen("4k3/8/8/8/8/8/4P3/3K4 w - - 0 1").unwrap();
    let mut second = Position::parse_fen("3k4/4p3/8/8/8/8/8/4K3 b - - 0 1").unwrap();
    let first_accumulator = network.build_accumulator(&first);
    let second_accumulator = network.build_accumulator(&second);
    assert_eq!(
        first_accumulator.sums[0],
        second_accumulator.sums[1],
    );
    assert_eq!(
        network.evaluate_head(
            &first_accumulator,
            Color::WHITE,
            0,
        ),
        network.evaluate_head(
            &second_accumulator,
            Color::BLACK,
            0,
        ),
    );
    assert_eq!(
        network.evaluate_policy(
            &first_accumulator,
            Color::WHITE,
            first.find_legal_move("e2e4").unwrap(),
        ),
        network.evaluate_policy(
            &second_accumulator,
            Color::BLACK,
            second.find_legal_move("e7e5").unwrap(),
        ),
    );
}

#[test]
fn test_value_only_training_preserves_bootstrap_policy() {
    let mut neural = create_test_network();
    neural.has_trained_policy = false;
    neural.has_trained_outcome = false;
    let trained = Network::create_neural_network(neural);
    let bootstrap = Network::create_network();
    let mut position = Position::parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
    let moves = position.generate_legal_moves(false).unwrap();

    for &chess_move in &moves.moves[..moves.count] {
        assert_eq!(
            trained.evaluate_policy(
                &position,
                &trained.build_accumulator(&position),
                chess_move,
            ),
            bootstrap.evaluate_policy(
                &position,
                &bootstrap.build_accumulator(&position),
                chess_move,
            ),
        );
    }
}

#[test]
fn test_policy_batch_matches_individual_heads() {
    let network = Network::create_neural_network(create_test_network());
    let mut position = Position::parse_fen("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1").unwrap();
    let moves = position.generate_legal_moves(false).unwrap();
    let accumulator = network.build_accumulator(&position);
    let policies = network.evaluate_policies(
        &position,
        &accumulator,
        &moves,
    );

    for (index, &chess_move) in moves.moves[..moves.count].iter().enumerate() {
        assert_eq!(
            policies[index],
            network.evaluate_policy(
                &position,
                &accumulator,
                chess_move,
            ),
        );
    }
}

#[test]
fn test_binary_neural_roundtrip_and_corruption_rejection() {
    let path = get_temporary_path("network.nnue");
    let original = Network::create_neural_network(create_test_network());
    original.save_network(&path).unwrap();
    let loaded = Network::load_network(&path).unwrap();
    assert_eq!(
        original.get_fingerprint(),
        loaded.get_fingerprint(),
    );
    let position = Position::parse_fen(START_FEN).unwrap();
    assert_eq!(
        original.evaluate_position(
            &position,
            &original.build_accumulator(&position),
        ),
        loaded.evaluate_position(
            &position,
            &loaded.build_accumulator(&position),
        ),
    );
    assert!(original.save_network(&path).is_err());
    let mut bytes = fs::read(&path).unwrap();
    bytes[100] ^= 1;
    fs::write(
        &path,
        &bytes,
    ).unwrap();
    assert!(Network::load_network(&path).is_err());
    bytes.truncate(100);
    fs::write(
        &path,
        &bytes,
    ).unwrap();
    assert!(Network::load_network(&path).is_err());
    fs::remove_file(path).unwrap();
}

#[test]
fn test_null_move_restores_history_hashes_and_en_passant() {
    let mut position = Position::parse_fen("4k3/8/8/3pP3/8/8/8/R3K3 w - d6 0 1").unwrap();
    let original_fen = position.get_fen();
    let original_key = position.get_key();
    let original_search_key = position.get_search_key();
    let original_history_count = position.get_history_count();
    let undo = position.make_null_move().unwrap();
    assert!(!position.is_rule_draw());
    assert!(position.has_valid_state());
    assert_eq!(
        position.get_side_to_move(),
        Color::BLACK,
    );
    position.unmake_move(&undo);
    assert_eq!(
        position.get_fen(),
        original_fen,
    );
    assert_eq!(
        position.get_key(),
        original_key,
    );
    assert_eq!(
        position.get_search_key(),
        original_search_key,
    );
    assert_eq!(
        position.get_history_count(),
        original_history_count,
    );

    let mut near_draw = Position::parse_fen("4k3/8/8/8/8/8/8/R3K3 w - - 99 1").unwrap();
    let undo = near_draw.make_null_move().unwrap();
    assert!(!near_draw.is_rule_draw());
    near_draw.unmake_move(&undo);
    assert_eq!(
        near_draw.get_halfmove_count(),
        99,
    );
}

#[test]
fn test_static_exchanges_include_pins_promotions_and_en_passant() {
    for (fen, move_name, expected_score) in [
        ("4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", 100,),
        ("3rk3/8/8/3p4/8/8/8/3QK3 w - - 0 1", "d1d5", -800,),
        ("4k3/4r3/3p4/8/8/8/8/3QR1K1 w - - 0 1", "d1d6", 100,),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", 800,),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 100,),
    ] {
        let mut position = Position::parse_fen(fen).unwrap();
        let chess_move = position.find_legal_move(move_name).unwrap();
        assert_eq!(
            calculate_exchange_score(
                &position,
                chess_move,
            ),
            expected_score,
        );
    }
}

#[test]
fn test_hybrid_verification_preserves_small_node_budgets() {
    for node_count in [1, 2, 5, 100, 500, 3000,] {
        let network = Network::create_network();
        let mut table = TranspositionTable::create_table(1).unwrap();
        let control = SearchControl::default();
        let limits = SearchLimits {
            node_count,
            ..SearchLimits::default()
        };
        control.start_search(&limits);
        let mut search = Search::create_search(
            &network,
            &mut table,
            &control,
            SearchOptions {
                policy_verification_depth_plies: 3,
                ..SearchOptions::default()
            },
        ).unwrap();
        let mut root = Position::parse_fen(START_FEN).unwrap();
        let searched = search.run_search(
            root.clone(),
            &limits,
            &mut |_| (),
        ).unwrap();
        assert!(searched.node_count <= node_count);
        assert_eq!(
            searched.best_move,
            root.find_legal_move(&searched.best_move.get_uci()).unwrap(),
        );
    }
}

#[test]
fn test_training_rejects_invalid_scores_unknown_outcomes_and_leaking_split() {
    let dataset_path = get_temporary_path("dataset.txt");
    let output_path = get_temporary_path("trained.nnue");

    for record in [
        format!("{START_FEN} | -2147483648 | ? | ? | game"),
        format!("{START_FEN} | 0 | 0.7 | ? | game"),
        format!("{START_FEN} | 0 | ? | e2e5:1 | game"),
        format!("{START_FEN} | 0 | ? | ? | one_game\n{START_FEN} | 0 | ? | ? | another_game"),
    ] {
        fs::write(
            &dataset_path,
            record + "\n",
        ).unwrap();
        assert!(train_network(
            &dataset_path,
            &output_path,
            &TrainingOptions {
                epoch_count: 1,
                ..TrainingOptions::default()
            },
            &mut |_| (),
        ).is_err());
        assert!(!output_path.exists());
    }

    fs::remove_file(dataset_path).unwrap();
}

#[test]
fn test_attack_features_change_when_a_slider_is_unblocked() {
    let mut position = Position::parse_fen("4k3/8/8/8/3rb3/3P4/3R4/4K3 w - - 0 1").unwrap();
    let network = create_test_network();
    let before = network.build_accumulator(&position);
    let chess_move = position.find_legal_move("d3e4").unwrap();
    let undo = position.make_move(chess_move).unwrap();
    let after = network.update_accumulator(
        &before,
        &undo,
    );
    assert_eq!(
        after.features,
        encode_features(&after.board),
    );
    assert_ne!(
        before.features,
        after.features,
    );
    assert_eq!(
        after.sums[0].len(),
        NEURAL_HIDDEN_COUNT,
    );
}

#[test]
fn test_quiescence_detects_stalemate_without_tactical_moves() {
    let network = Network::create_network();
    let mut table = TranspositionTable::create_table(1).unwrap();
    let control = SearchControl::default();
    let mut root = Position::parse_fen("7k/5Q2/5K2/8/8/8/8/8 w - - 0 1").unwrap();
    let limits = SearchLimits {
        depth_plies: 1,
        root_moves: vec![root.find_legal_move("f7g6").unwrap(); 1],
        ..SearchLimits::default()
    };
    control.start_search(&limits);
    let mut search = Search::create_search(
        &network,
        &mut table,
        &control,
        SearchOptions {
            puct_simulation_count: 0,
            ..SearchOptions::default()
        },
    ).unwrap();
    let searched = search.run_search(
        root,
        &limits,
        &mut |_| (),
    ).unwrap();
    assert_eq!(
        searched.score,
        0,
    );
}

#[test]
fn test_root_verification_matches_full_root_search() {
    for fen in [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let mut scores = Vec::new();

        for variation_count in [1, 3,] {
            let network = Network::create_network();
            let mut table = TranspositionTable::create_table(1).unwrap();
            let control = SearchControl::default();
            let limits = SearchLimits {
                depth_plies: 3,
                ..SearchLimits::default()
            };
            control.start_search(&limits);
            let mut search = Search::create_search(
                &network,
                &mut table,
                &control,
                SearchOptions {
                    variation_count,
                    puct_simulation_count: 0,
                    aggression_margin_centipawns: 0,
                    is_selective_search_enabled: false,
                    ..SearchOptions::default()
                },
            ).unwrap();
            let searched = search.run_search(
                Position::parse_fen(fen).unwrap(),
                &limits,
                &mut |_| (),
            ).unwrap();
            scores.push(searched.score);
        }

        assert_eq!(
            scores[0],
            scores[1],
        );
    }
}
