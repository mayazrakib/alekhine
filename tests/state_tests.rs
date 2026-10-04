use std::sync::atomic::Ordering;

use alekhine::network::Network;
use alekhine::position::Position;
use alekhine::search::{
    Search,
    SearchControl,
    SearchLimits,
    SearchOptions,
};
use alekhine::transposition_table::TranspositionTable;
use alekhine::types::{
    MATE_SCORE,
    MAXIMUM_HISTORY_COUNT,
    Move,
    START_FEN,
};

#[test]
fn test_castling_moves_relocate_both_pieces_and_restore_all_state() {
    for (side, move_name, expected_placement, expected_rights, expected_fullmove) in [
        ("w", "e1g1", "r3k2r/8/8/8/8/8/8/R4RK1", "kq", "1",),
        ("w", "e1c1", "r3k2r/8/8/8/8/8/8/2KR3R", "kq", "1",),
        ("b", "e8g8", "r4rk1/8/8/8/8/8/8/R3K2R", "KQ", "2",),
        ("b", "e8c8", "2kr3r/8/8/8/8/8/8/R3K2R", "KQ", "2",),
    ] {
        let original_fen = format!("r3k2r/8/8/8/8/8/8/R3K2R {side} KQkq - 0 1");
        let mut position = Position::parse_fen(&original_fen).unwrap();
        let original_key = position.get_key();
        let original_search_key = position.get_search_key();
        let chess_move = position.find_legal_move(move_name).unwrap();
        let undo = position.make_move(chess_move).unwrap();
        let child_fen = position.get_fen();
        let fields: Vec<&str> = child_fen.split_whitespace().collect();
        assert_eq!(
            fields[0],
            expected_placement,
        );
        assert_eq!(
            fields[2],
            expected_rights,
        );
        assert_eq!(
            fields[5],
            expected_fullmove,
        );
        assert!(position.has_valid_state());
        position.unmake_move(&undo);
        assert_eq!(
            position.get_fen(),
            original_fen,
        );
        assert_eq!(
            (position.get_key(), position.get_search_key(),),
            (original_key, original_search_key,),
        );
    }
}

#[test]
fn test_rook_moves_and_corner_captures_update_both_castling_rights() {
    for (side, move_name, expected_rights) in [("w", "h1h8", "Qq",), ("w", "a1a8", "Kk",), ("b", "h8h1", "Qq",), ("b", "a8a1", "Kk",),] {
        let mut position = Position::parse_fen(&format!("r3k2r/8/8/8/8/8/8/R3K2R {side} KQkq - 0 1")).unwrap();
        let chess_move = position.find_legal_move(move_name).unwrap();
        position.make_move(chess_move).unwrap();
        assert_eq!(
            position.get_fen().split_whitespace().nth(2).unwrap(),
            expected_rights,
        );
        assert_eq!(
            position.get_halfmove_count(),
            0,
        );
    }
}

#[test]
fn test_move_counters_reset_on_pawns_and_captures() {
    let mut position = Position::parse_fen(START_FEN).unwrap();

    for (move_name, expected_halfmoves, expected_fullmove) in [("g1f3", 1, "1",), ("g8f6", 2, "2",), ("e2e4", 0, "2",), ("d7d5", 0, "3",), ("e4d5", 0, "3",)] {
        let chess_move = position.find_legal_move(move_name).unwrap();
        position.make_move(chess_move).unwrap();
        assert_eq!(
            position.get_halfmove_count(),
            expected_halfmoves,
        );
        assert_eq!(
            position.get_fen().split_whitespace().nth(5).unwrap(),
            expected_fullmove,
        );
    }
}

#[test]
fn test_en_passant_hash_only_changes_for_legal_captures() {
    for (placement, has_legal_capture) in [
        ("4k3/8/8/3p4/8/8/8/R3K3", false,),
        ("4k3/8/8/3pP3/8/8/8/R3K3", true,),
        ("k3r3/8/8/3pP3/8/8/8/4K3", false,),
    ] {
        let with_target = Position::parse_fen(&format!("{placement} w - d6 0 1")).unwrap();
        let without_target = Position::parse_fen(&format!("{placement} w - - 0 1")).unwrap();
        assert_eq!(
            with_target.get_key() != without_target.get_key(),
            has_legal_capture,
        );
    }
}

#[test]
fn test_pawn_hash_ignores_quiet_piece_moves_and_tracks_pawn_moves() {
    let mut position = Position::parse_fen(START_FEN).unwrap();
    let original_pawn_key = position.get_pawn_key();

    for move_name in ["g1f3", "g8f6",] {
        let chess_move = position.find_legal_move(move_name).unwrap();
        position.make_move(chess_move).unwrap();
        assert_eq!(
            position.get_pawn_key(),
            original_pawn_key,
        );
    }

    let chess_move = position.find_legal_move("e2e4").unwrap();
    let undo = position.make_move(chess_move).unwrap();
    assert_ne!(
        position.get_pawn_key(),
        original_pawn_key,
    );
    position.unmake_move(&undo);
    assert_eq!(
        position.get_pawn_key(),
        original_pawn_key,
    );
}

#[test]
fn test_tactical_move_generation_matches_filtering_full_legal_moves() {
    for fen in [START_FEN, "4k3/P7/8/3pP3/8/8/8/4K3 w - d6 0 1", "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",] {
        let mut position = Position::parse_fen(fen).unwrap();
        let original_fen = position.get_fen();
        let original_key = position.get_key();
        let full_moves = position.generate_legal_moves(false).unwrap();
        let tactical_moves = position.generate_legal_moves(true).unwrap();
        let mut expected: Vec<String> = full_moves.moves[..full_moves.count].iter()
            .filter(|chess_move| chess_move.is_tactical()).map(|chess_move| chess_move.get_uci()).collect();
        let mut actual: Vec<String> = tactical_moves.moves[..tactical_moves.count].iter().map(|chess_move| chess_move.get_uci()).collect();
        expected.sort();
        actual.sort();
        assert_eq!(
            actual,
            expected,
        );
        assert_eq!(
            position.has_legal_move().unwrap(),
            full_moves.count != 0,
        );
        assert_eq!(
            (position.get_fen(), position.get_key(),),
            (original_fen, original_key,),
        );
    }
}

#[test]
fn test_random_move_stacks_unwind_to_every_prior_state() {
    let network = Network::create_network();
    let mut position = Position::parse_fen(START_FEN).unwrap();
    let mut accumulator = network.build_accumulator(&position);
    let mut history = Vec::new();
    let mut seed = 20261005u64;

    for _ in 0..256 {
        let moves = position.generate_legal_moves(false).unwrap();

        if moves.count == 0 {
            break;
        }

        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let chess_move = moves.moves[(seed % moves.count as u64) as usize];
        let previous_state = (position.get_fen(), position.get_key(), position.get_search_key(), accumulator,);
        let undo = position.make_move(chess_move).unwrap();
        accumulator = network.update_accumulator(
            &accumulator,
            &undo,
        );
        history.push((undo, previous_state,));
        assert!(position.has_valid_state());
    }

    while let Some((undo, previous_state)) = history.pop() {
        position.unmake_move(&undo);
        accumulator = network.build_accumulator(&position);
        assert_eq!(
            (position.get_fen(), position.get_key(), position.get_search_key(), accumulator,),
            previous_state,
        );
        assert!(position.has_valid_state());
    }

    assert_eq!(
        position.get_fen(),
        START_FEN,
    );
}

#[test]
fn test_history_capacity_failure_preserves_position() {
    let mut position = Position::parse_fen(START_FEN).unwrap();
    let move_names = ["g1f3", "g8f6", "f3g1", "f6g8",];

    for index in 0..MAXIMUM_HISTORY_COUNT - 1 {
        let chess_move = position.find_legal_move(move_names[index % move_names.len()]).unwrap();
        position.make_move(chess_move).unwrap();
    }

    let original_fen = position.get_fen();
    let original_key = position.get_key();
    let chess_move = Move::create_move(
        45,
        62,
        alekhine::types::PieceType::NONE,
        alekhine::types::MoveFlag::QUIET,
    );
    assert!(position.make_move(chess_move).is_err());
    assert!(position.make_null_move().is_err());
    assert_eq!(
        (position.get_fen(), position.get_key(), position.get_history_count(),),
        (original_fen, original_key, MAXIMUM_HISTORY_COUNT,),
    );
}

#[test]
fn test_search_can_be_cancelled_by_report_observer() {
    let network = Network::create_network();
    let mut table = TranspositionTable::create_table(1).unwrap();
    let control = SearchControl::default();
    let limits = SearchLimits {
        depth_plies: 10,
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
    let mut reports = Vec::new();
    let searched = search.run_search(
        Position::parse_fen(START_FEN).unwrap(),
        &limits,
        &mut |report| {
            reports.push(report.depth_plies);
            control.should_stop.store(
                true,
                Ordering::SeqCst,
            );
        },
    ).unwrap();
    assert_eq!(
        reports,
        vec![1,],
    );
    assert_eq!(
        searched.completed_depth_plies,
        1,
    );
    assert_ne!(
        searched.best_move,
        Move::default(),
    );
}

#[test]
fn test_search_rejects_invalid_option_boundaries() {
    let network = Network::create_network();
    let mut table = TranspositionTable::create_table(1).unwrap();
    let control = SearchControl::default();

    for options in [
        SearchOptions {
            variation_count: 0,
            ..SearchOptions::default()
        },
        SearchOptions {
            variation_count: 17,
            ..SearchOptions::default()
        },
        SearchOptions {
            puct_simulation_count: 4097,
            ..SearchOptions::default()
        },
        SearchOptions {
            policy_verification_depth_plies: -1,
            ..SearchOptions::default()
        },
        SearchOptions {
            policy_verification_depth_plies: 4,
            ..SearchOptions::default()
        },
        SearchOptions {
            aggression_margin_centipawns: -1,
            ..SearchOptions::default()
        },
        SearchOptions {
            aggression_margin_centipawns: 101,
            ..SearchOptions::default()
        },
        SearchOptions {
            policy_uncertainty_percent: 101,
            ..SearchOptions::default()
        },
    ] {
        assert!(Search::create_search(
            &network,
            &mut table,
            &control,
            options,
        ).is_err());
    }
}

#[test]
fn test_search_ablations_preserve_legal_variations_and_node_limits() {
    for is_selective_search_enabled in [false, true,] {
        for puct_simulation_count in [0, 16,] {
            let network = Network::create_network();
            let mut table = TranspositionTable::create_table(1).unwrap();
            let control = SearchControl::default();
            let limits = SearchLimits {
                depth_plies: 3,
                node_count: 10000,
                ..SearchLimits::default()
            };
            control.start_search(&limits);
            let mut search = Search::create_search(
                &network,
                &mut table,
                &control,
                SearchOptions {
                    is_selective_search_enabled,
                    puct_simulation_count,
                    variation_count: 3,
                    aggression_margin_centipawns: 100,
                    policy_uncertainty_percent: 100,
                    ..SearchOptions::default()
                },
            ).unwrap();
            let searched = search.run_search(
                Position::parse_fen(START_FEN).unwrap(),
                &limits,
                &mut |report| {
                    let mut position = Position::parse_fen(START_FEN).unwrap();

                    for &chess_move in &report.principal_variation {
                        assert_eq!(
                            position.find_legal_move(&chess_move.get_uci()).unwrap(),
                            chess_move,
                        );
                        position.make_move(chess_move).unwrap();
                    }
                },
            ).unwrap();
            assert!(searched.node_count <= limits.node_count);
            assert_ne!(
                searched.best_move,
                Move::default(),
            );
        }
    }
}

#[test]
fn test_null_subtree_suppresses_repetition_and_restores_draw_after_unmake() {
    let mut position = Position::parse_fen(START_FEN).unwrap();

    for move_name in ["g1f3", "g8f6", "f3g1", "f6g8",].repeat(2) {
        let chess_move = position.find_legal_move(move_name).unwrap();
        position.make_move(chess_move).unwrap();
    }

    assert!(position.is_rule_draw());
    let original_fen = position.get_fen();
    let first_null = position.make_null_move().unwrap();
    let second_null = position.make_null_move().unwrap();
    assert!(!position.is_repetition(3));
    assert!(!position.is_rule_draw());
    position.unmake_move(&second_null);
    position.unmake_move(&first_null);
    assert!(position.is_rule_draw());
    assert_eq!(
        position.get_fen(),
        original_fen,
    );
}

#[test]
fn test_aggression_and_policy_exploration_preserve_forced_mate_priority() {
    for aggression_margin_centipawns in [0, 100,] {
        for puct_simulation_count in [0, 16,] {
            let network = Network::create_network();
            let mut table = TranspositionTable::create_table(1).unwrap();
            let control = SearchControl::default();
            let limits = SearchLimits {
                depth_plies: 4,
                node_count: 10000,
                ..SearchLimits::default()
            };
            control.start_search(&limits);
            let mut search = Search::create_search(
                &network,
                &mut table,
                &control,
                SearchOptions {
                    aggression_margin_centipawns,
                    puct_simulation_count,
                    policy_uncertainty_percent: 100,
                    ..SearchOptions::default()
                },
            ).unwrap();
            let mut root = Position::parse_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap();
            let searched = search.run_search(
                root.clone(),
                &limits,
                &mut |_| (),
            ).unwrap();
            assert_eq!(
                searched.score,
                MATE_SCORE - 1,
            );
            root.make_move(searched.best_move).unwrap();
            assert!(root.is_in_check(root.get_side_to_move()));
            assert!(!root.has_legal_move().unwrap());
        }
    }
}
