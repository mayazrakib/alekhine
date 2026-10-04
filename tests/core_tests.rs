use std::fs;
use std::sync::atomic::{
    AtomicU64,
    Ordering,
};

use alekhine::network::{
    compute_outcome_probabilities,
    Network,
};
use alekhine::position::{
    count_perft,
    Position,
};
use alekhine::search::{
    calculate_time_budget,
    Search,
    SearchControl,
    SearchLimits,
    SearchOptions,
    SearchReport,
    SearchResult,
};
use alekhine::transposition_table::{
    BoundType,
    TranspositionTable,
};
use alekhine::types::{
    Color,
    MATE_SCORE,
    Move,
    MoveFlag,
    PieceType,
    START_FEN,
};

const KIWIPETE_FEN: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
static NETWORK_TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn require_perft(
    fen: &str,
    depth_plies: i32,
    expected_count: u64,
) {
    let mut position = Position::parse_fen(fen).unwrap();
    let original_fen = position.get_fen();
    let node_count = count_perft(
        &mut position,
        depth_plies,
    ).unwrap();
    assert_eq!(
        node_count,
        expected_count,
    );
    assert!(position.has_valid_state());
    assert_eq!(
        position.get_fen(),
        original_fen,
    );
}

#[test]
fn test_start_position_perft() {
    require_perft(
        START_FEN,
        4,
        197281,
    );
}

#[test]
fn test_kiwipete_perft() {
    require_perft(
        KIWIPETE_FEN,
        4,
        4085603,
    );
}

#[test]
fn test_en_passant_perft() {
    require_perft(
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        4,
        43238,
    );
}

#[test]
fn test_promotion_and_check_evasion_perft() {
    require_perft(
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        3,
        9467,
    );
}

#[test]
fn test_underpromotion_perft() {
    require_perft(
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        3,
        62379,
    );
}

#[test]
fn test_tactical_position_perft() {
    require_perft(
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        3,
        89890,
    );
}

#[test]
fn test_make_unmake_and_accumulator_invariants() {
    let network = Network::create_network();
    let mut seed = 20261004u64;
    let mut position = Position::parse_fen(KIWIPETE_FEN).unwrap();
    let original_key = position.get_key();
    let original_fen = position.get_fen();
    let mut undos = Vec::new();
    let mut accumulator = network.build_accumulator(&position);

    for _ in 0..200 {
        let moves = position.generate_legal_moves(false).unwrap();

        if moves.count == 0 {
            break;
        }

        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let index = (seed % moves.count as u64) as usize;
        let undo = position.make_move(moves.moves[index]).unwrap();
        accumulator = network.update_accumulator(
            &accumulator,
            &undo,
        );
        undos.push(undo);
        assert!(position.has_valid_state());
        assert_eq!(
            accumulator,
            network.build_accumulator(&position),
        );
    }

    for undo in undos.iter().rev() {
        position.unmake_move(undo);
        assert!(position.has_valid_state());
    }

    assert_eq!(
        position.get_key(),
        original_key,
    );
    assert_eq!(
        position.get_fen(),
        original_fen,
    );
}

#[test]
fn test_special_move_accumulators() {
    let network = Network::create_network();

    for fen in [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ] {
        let mut position = Position::parse_fen(fen).unwrap();
        let original_fen = position.get_fen();
        let accumulator = network.build_accumulator(&position);
        let moves = position.generate_legal_moves(false).unwrap();

        for &chess_move in &moves.moves[..moves.count] {
            let undo = position.make_move(chess_move).unwrap();
            assert_eq!(
                network.update_accumulator(
                    &accumulator,
                    &undo,
                ),
                network.build_accumulator(&position),
            );
            assert!(position.has_valid_state());
            position.unmake_move(&undo);
            assert_eq!(
                position.get_fen(),
                original_fen,
            );
        }
    }
}

#[test]
fn test_fen_roundtrip() {
    for fen in [START_FEN, KIWIPETE_FEN, "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 7",] {
        assert_eq!(
            Position::parse_fen(fen).unwrap().get_fen(),
            fen,
        );
    }
}

#[test]
fn test_invalid_fen_rejection() {
    for fen in [
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w K - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - e6 0 1",
        "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - -1 1",
        "4k3/8/8/8/8/8/4K3/8 w - - 0 1 extra",
        "8/8/8/8/8/8/4k3/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/P3K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w KK - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - h1 0 1",
        "4k3/8/8/8/8/P7/PPPPPPPP/4K3 w - - 0 1",
    ] {
        assert!(Position::parse_fen(fen).is_err());
    }
}

#[test]
fn test_pinned_en_passant_rejection() {
    let mut position = Position::parse_fen("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
    let moves = position.generate_legal_moves(false).unwrap();

    for chess_move in &moves.moves[..moves.count] {
        assert_ne!(
            chess_move.get_uci(),
            "e5d6",
        );
    }

    let without_en_passant = Position::parse_fen("k3r3/8/8/3pP3/8/8/8/4K3 w - - 0 1").unwrap();
    assert_eq!(
        position.get_key(),
        without_en_passant.get_key(),
    );
}

#[test]
fn test_castling_through_check_rejection() {
    let mut position = Position::parse_fen("4kr2/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
    let moves = position.generate_legal_moves(false).unwrap();

    for chess_move in &moves.moves[..moves.count] {
        assert_ne!(
            chess_move.get_uci(),
            "e1g1",
        );
    }
}

#[test]
fn test_threefold_repetition() {
    let mut position = Position::parse_fen(START_FEN).unwrap();

    for cycle in 0..2 {
        for move_name in ["g1f3", "g8f6", "f3g1", "f6g8",] {
            let chess_move = position.find_legal_move(move_name).unwrap();
            position.make_move(chess_move).unwrap();
        }

        assert_eq!(
            position.is_repetition(3),
            cycle == 1,
        );
    }
}

#[test]
fn test_search_hash_distinguishes_draw_context() {
    let mut repeated = Position::parse_fen(START_FEN).unwrap();

    for move_name in ["g1f3", "g8f6", "f3g1", "f6g8",] {
        let chess_move = repeated.find_legal_move(move_name).unwrap();
        repeated.make_move(chess_move).unwrap();
    }

    let standalone = Position::parse_fen(&repeated.get_fen()).unwrap();
    assert_eq!(
        repeated.get_key(),
        standalone.get_key(),
    );
    assert_ne!(
        repeated.get_search_key(),
        standalone.get_search_key(),
    );
    let fresh = Position::parse_fen(START_FEN).unwrap();
    assert_ne!(
        repeated.get_search_key(),
        fresh.get_search_key(),
    );
}

#[test]
fn test_fifty_move_rule() {
    let position = Position::parse_fen("4k3/8/8/8/8/8/8/R3K3 w - - 100 51").unwrap();
    assert!(position.is_rule_draw());
}

#[test]
fn test_insufficient_material() {
    for fen in [
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/1N2K3 w - - 0 1",
        "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1",
    ] {
        assert!(Position::parse_fen(fen).unwrap().has_insufficient_material());
    }

    assert!(!Position::parse_fen("4k3/8/8/8/8/8/8/1NN1K3 w - - 0 1").unwrap().has_insufficient_material());
}

#[test]
fn test_transposition_bounds_and_mate_normalization() {
    let mut table = TranspositionTable::create_table(1).unwrap();
    table.start_generation();
    let best_move = Move::create_move(
        12,
        28,
        PieceType::NONE,
        MoveFlag::QUIET,
    );
    table.store_position(
        42,
        best_move,
        MATE_SCORE - 7,
        12,
        BoundType::LOWER,
        3,
    );
    let entry = table.probe_position(42).unwrap();
    assert_eq!(
        entry.best_move,
        best_move,
    );
    assert_eq!(
        entry.bound,
        BoundType::LOWER,
    );
    assert_eq!(
        TranspositionTable::decode_score(
            entry.score as i32,
            5,
        ),
        MATE_SCORE - 9,
    );
    table.store_position(
        43,
        best_move,
        -MATE_SCORE + 7,
        12,
        BoundType::EXACT,
        3,
    );
    assert_eq!(
        TranspositionTable::decode_score(
            table.probe_position(43).unwrap().score as i32,
            5,
        ),
        -MATE_SCORE + 9,
    );
    assert!(table.probe_position(44).is_none());
    table.clear_table();
    assert!(table.probe_position(42).is_none());
}

#[test]
fn test_network_roundtrip_and_corruption_rejection() {
    let sequence = NETWORK_TEST_SEQUENCE.fetch_add(
        1,
        Ordering::SeqCst,
    );
    let path = std::env::temp_dir().join(format!(
        "alekhine_test_{}_{sequence}.network",
        std::process::id(),
    ));
    let original = Network::create_network();
    original.save_network(&path).unwrap();
    let restored = Network::load_network(&path).unwrap();
    assert_eq!(
        original.get_fingerprint(),
        restored.get_fingerprint(),
    );
    assert!(!restored.is_trained());
    let mut payload = fs::read_to_string(&path).unwrap();
    payload.push_str("Corrupted payload.\n");
    fs::write(
        &path,
        payload,
    ).unwrap();
    let is_rejected = Network::load_network(&path).is_err();
    fs::remove_file(path).unwrap();
    assert!(is_rejected);
}

fn run_test_search(
    position: Position,
    limits: &SearchLimits,
    options: SearchOptions,
    observer: &mut dyn FnMut(&SearchReport),
) -> SearchResult {
    let network = Network::create_network();
    let mut table = TranspositionTable::create_table(1).unwrap();
    let control = SearchControl::default();
    control.start_search(limits);
    let mut search = Search::create_search(
        &network,
        &mut table,
        &control,
        options,
    ).unwrap();

    return search.run_search(
        position,
        limits,
        observer,
    ).unwrap();
}

#[test]
fn test_search_finds_mate_in_one() {
    let mut position = Position::parse_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap();
    let limits = SearchLimits {
        depth_plies: 2,
        ..SearchLimits::default()
    };
    let search_result = run_test_search(
        position.clone(),
        &limits,
        SearchOptions::default(),
        &mut |_| (),
    );
    position.make_move(search_result.best_move).unwrap();
    assert_eq!(
        search_result.score,
        MATE_SCORE - 1,
    );
    assert_eq!(
        position.generate_legal_moves(false).unwrap().count,
        0,
    );
    assert!(position.is_in_check(Color::BLACK));
}

#[test]
fn test_checkmate_precedes_fifty_move_draw() {
    let position = Position::parse_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 100 51").unwrap();
    let limits = SearchLimits {
        depth_plies: 1,
        ..SearchLimits::default()
    };
    let search_result = run_test_search(
        position,
        &limits,
        SearchOptions::default(),
        &mut |_| (),
    );
    assert_eq!(
        search_result.best_move.encoding,
        0,
    );
    assert_eq!(
        search_result.score,
        -MATE_SCORE,
    );
}

#[test]
fn test_stalemate_has_no_best_move() {
    let limits = SearchLimits {
        depth_plies: 2,
        ..SearchLimits::default()
    };
    let search_result = run_test_search(
        Position::parse_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap(),
        &limits,
        SearchOptions::default(),
        &mut |_| (),
    );
    assert_eq!(
        search_result.best_move.encoding,
        0,
    );
    assert_eq!(
        search_result.score,
        0,
    );
}

#[test]
fn test_node_limit_and_legal_fallback() {
    let mut position = Position::parse_fen(START_FEN).unwrap();
    let limits = SearchLimits {
        node_count: 1,
        ..SearchLimits::default()
    };
    let search_result = run_test_search(
        position.clone(),
        &limits,
        SearchOptions::default(),
        &mut |_| (),
    );
    assert!(search_result.node_count <= 1);
    assert_eq!(
        search_result.best_move,
        position.find_legal_move(&search_result.best_move.get_uci()).unwrap(),
    );
}

#[test]
fn test_search_is_reproducible() {
    let limits = SearchLimits {
        depth_plies: 3,
        ..SearchLimits::default()
    };
    let first = run_test_search(
        Position::parse_fen(START_FEN).unwrap(),
        &limits,
        SearchOptions::default(),
        &mut |_| (),
    );
    let second = run_test_search(
        Position::parse_fen(START_FEN).unwrap(),
        &limits,
        SearchOptions::default(),
        &mut |_| (),
    );
    assert_eq!(
        first.best_move,
        second.best_move,
    );
    assert_eq!(
        first.score,
        second.score,
    );
    assert_eq!(
        first.node_count,
        second.node_count,
    );
    assert_eq!(
        first.principal_variation,
        second.principal_variation,
    );
}

#[test]
fn test_principal_variations_are_legal() {
    let limits = SearchLimits {
        depth_plies: 3,
        ..SearchLimits::default()
    };
    let options = SearchOptions {
        variation_count: 3,
        ..SearchOptions::default()
    };
    let mut report_count = 0;
    run_test_search(
        Position::parse_fen(KIWIPETE_FEN).unwrap(),
        &limits,
        options,
        &mut |report| {
            let mut position = Position::parse_fen(KIWIPETE_FEN).unwrap();

            for &chess_move in &report.principal_variation {
                assert_eq!(
                    position.find_legal_move(&chess_move.get_uci()).unwrap(),
                    chess_move,
                );
                position.make_move(chess_move).unwrap();
            }

            report_count += 1;
        },
    );
    assert_eq!(
        report_count,
        9,
    );
}

#[test]
fn test_root_move_restriction() {
    let mut position = Position::parse_fen(START_FEN).unwrap();
    let limits = SearchLimits {
        depth_plies: 2,
        root_moves: vec![position.find_legal_move("e2e4").unwrap(),],
        ..SearchLimits::default()
    };
    let search_result = run_test_search(
        position,
        &limits,
        SearchOptions::default(),
        &mut |_| (),
    );
    assert_eq!(
        search_result.best_move.get_uci(),
        "e2e4",
    );
}

#[test]
fn test_time_budget_preserves_reserve() {
    let budget = calculate_time_budget(
        1000,
        100,
        0,
        20,
    );
    assert!(budget.optimum_milliseconds > 0);
    assert!(budget.maximum_milliseconds >= budget.optimum_milliseconds);
    assert!(budget.maximum_milliseconds <= 980);
}

#[test]
fn test_outcome_probabilities_are_normalized() {
    for score in [-20000, -500, 0, 500, 20000,] {
        let probabilities = compute_outcome_probabilities(score);
        assert!(probabilities.win >= 0.0);
        assert!(probabilities.draw >= 0.0);
        assert!(probabilities.loss >= 0.0);
        assert!((probabilities.win + probabilities.draw + probabilities.loss - 1.0).abs() < 0.000001);
    }
}
