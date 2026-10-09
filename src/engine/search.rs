use std::cmp::Reverse;
use std::sync::atomic::{
    AtomicBool,
    AtomicI64,
    Ordering,
};
use std::sync::OnceLock;
use std::time::Instant;

use crate::evaluation::network::{
    Accumulator,
    Network,
};
use crate::chess::position::Position;
use crate::chess::tactical::{
    calculate_attack_priority,
    calculate_exchange_score,
};
use crate::engine::transposition_table::{
    BoundType,
    TranspositionTable,
};
use crate::chess::types::{
    BOARD_SQUARE_COUNT,
    EngineError,
    EngineResult,
    get_color_index,
    get_piece_index,
    get_piece_score,
    INFINITY_SCORE,
    MATE_SCORE,
    MATE_THRESHOLD,
    MAXIMUM_HISTORY_COUNT,
    MAXIMUM_MOVE_COUNT,
    MAXIMUM_PLY_COUNT,
    Move,
    MoveFlag,
    MoveList,
    PIECE_SCORES,
};

const MAXIMUM_PUCT_NODE_COUNT: usize = 8192;
const MAXIMUM_PUCT_EDGE_COUNT: usize = 131072;
const PUCT_EXPLORATION_FACTOR: f64 = 1.5;
const ASPIRATION_WINDOW_SCORE: i32 = 30;
const HISTORY_LIMIT: i32 = 16000;
const CONTINUATION_CONTEXT_COUNT: usize = 12 * BOARD_SQUARE_COUNT;
const CORRECTION_ENTRY_COUNT: usize = 16384;
const MAXIMUM_EXTENSION_COUNT: usize = 8;
const POLICY_BUDGET_DIVISOR: u64 = 5;
const POLICY_NODE_ALLOWANCE_PER_SIMULATION: u64 = 16;
const POLICY_VARIANCE_FLOOR: f64 = 0.01;
const REVERSE_FUTILITY_MARGIN_CENTIPAWNS_PER_PLY: i32 = 100;
const SINGULAR_MARGIN_CENTIPAWNS_PER_PLY: i32 = 20;
static CLOCK_EPOCH: OnceLock<Instant> = OnceLock::new();

#[derive(
    Clone,
    Copy,
    Default,
)]
pub struct TimeBudget {
    pub optimum_milliseconds: i64,
    pub maximum_milliseconds: i64,
}

#[derive(Clone)]
pub struct SearchLimits {
    pub depth_plies: i32,
    pub node_count: u64,
    pub time: TimeBudget,
    pub is_infinite: bool,
    pub is_pondering: bool,
    pub root_moves: Vec<Move>,
}

impl Default for SearchLimits {
    fn default() -> Self {
        return Self {
            depth_plies: MAXIMUM_PLY_COUNT as i32 - 2,
            node_count: 0,
            time: TimeBudget::default(),
            is_infinite: false,
            is_pondering: false,
            root_moves: Vec::new(),
        };
    }
}

#[derive(
    Clone,
    Copy,
)]
pub struct SearchOptions {
    pub variation_count: usize,
    pub puct_simulation_count: usize,
    pub policy_verification_depth_plies: i32,
    pub aggression_margin_centipawns: i32,
    pub is_selective_search_enabled: bool,
    pub policy_uncertainty_percent: i32,
}

impl Default for SearchOptions {
    fn default() -> Self {
        return Self {
            variation_count: 1,
            puct_simulation_count: 128,
            policy_verification_depth_plies: 1,
            aggression_margin_centipawns: 15,
            is_selective_search_enabled: true,
            policy_uncertainty_percent: 25,
        };
    }
}

#[derive(Default)]
pub struct SearchControl {
    pub should_stop: AtomicBool,
    pub is_pondering: AtomicBool,
    pub hard_deadline_milliseconds: AtomicI64,
    pub soft_deadline_milliseconds: AtomicI64,
}

#[derive(
    Clone,
    Debug,
)]
pub struct SearchReport {
    pub depth_plies: i32,
    pub selective_depth_plies: usize,
    pub score: i32,
    pub variation_index: usize,
    pub hashfull_per_mille: i32,
    pub node_count: u64,
    pub elapsed_milliseconds: i64,
    pub principal_variation: Vec<Move>,
}

#[derive(
    Clone,
    Debug,
    Default,
)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
    pub completed_depth_plies: i32,
    pub node_count: u64,
    pub principal_variation: Vec<Move>,
}

#[derive(Default)]
struct PolicyNode {
    first_edge: usize,
    edge_count: usize,
    visit_count: u32,
    value_sum: f64,
    is_expanded: bool,
    is_terminal: bool,
    terminal_value: f64,
}

struct PolicyEdge {
    chess_move: Move,
    child_index: Option<usize>,
    visit_count: u32,
    prior: f64,
    value_sum: f64,
    value_squared_sum: f64,
}

struct RootVariation {
    chess_move: Move,
    score: i32,
    principal_variation: Vec<Move>,
}

pub fn get_monotonic_milliseconds() -> i64 {
    return CLOCK_EPOCH.get_or_init(Instant::now).elapsed().as_millis() as i64 + 1;
}

pub fn calculate_time_budget(
    remaining_milliseconds: i64,
    increment_milliseconds: i64,
    moves_to_go: i32,
    overhead_milliseconds: i64,
) -> TimeBudget {
    let available_milliseconds = (remaining_milliseconds - overhead_milliseconds).max(1);

    let move_count = if moves_to_go > 0 {
        moves_to_go
    } else {
        30
    } as i64;

    let optimum_milliseconds = ((available_milliseconds / move_count) + ((increment_milliseconds * 3) / 4)).clamp(
        1,
        available_milliseconds,
    );
    let maximum_milliseconds = available_milliseconds.min(optimum_milliseconds.max((optimum_milliseconds * 4).min(available_milliseconds / 3)));

    return TimeBudget {
        optimum_milliseconds,
        maximum_milliseconds,
    };
}

impl SearchControl {
    pub fn activate_clock(
        &self,
        budget: TimeBudget,
    ) {
        let now_milliseconds = get_monotonic_milliseconds();
        self.hard_deadline_milliseconds.store(

            if budget.maximum_milliseconds > 0 {
                now_milliseconds + budget.maximum_milliseconds
            } else {
                0
            },

            Ordering::SeqCst,
        );
        self.soft_deadline_milliseconds.store(

            if budget.optimum_milliseconds > 0 {
                now_milliseconds + budget.optimum_milliseconds
            } else {
                0
            },

            Ordering::SeqCst,
        );
        self.is_pondering.store(
            false,
            Ordering::SeqCst,
        );
    }

    pub fn start_search(
        &self,
        limits: &SearchLimits,
    ) {
        self.should_stop.store(
            false,
            Ordering::SeqCst,
        );
        self.hard_deadline_milliseconds.store(
            0,
            Ordering::SeqCst,
        );
        self.soft_deadline_milliseconds.store(
            0,
            Ordering::SeqCst,
        );
        self.is_pondering.store(
            limits.is_pondering,
            Ordering::SeqCst,
        );

        if !limits.is_pondering && !limits.is_infinite {
            self.activate_clock(limits.time);
        }
    }
}

pub struct Search<'a> {
    network: &'a Network,
    transposition_table: &'a mut TranspositionTable,
    control: &'a SearchControl,
    options: SearchOptions,
    limits: SearchLimits,
    started_at: Instant,
    node_count: u64,
    selective_depth_plies: usize,
    was_interrupted: bool,
    variations: [[Move; MAXIMUM_PLY_COUNT]; MAXIMUM_PLY_COUNT],
    variation_lengths: [usize; MAXIMUM_PLY_COUNT],
    killers: [[Move; 2]; MAXIMUM_PLY_COUNT],
    quiet_history: [[[i32; BOARD_SQUARE_COUNT]; BOARD_SQUARE_COUNT]; 2],
    capture_history: [[[i16; 7]; BOARD_SQUARE_COUNT]; 12],
    continuation_history: Vec<i16>,
    correction_history: [[i16; CORRECTION_ENTRY_COUNT]; 2],
    previous_contexts: [Option<usize>; MAXIMUM_PLY_COUNT],
    previous_moves: [Move; MAXIMUM_PLY_COUNT],
    extension_counts: [usize; MAXIMUM_PLY_COUNT],
    excluded_moves: [Move; MAXIMUM_PLY_COUNT],
    is_null_verification: bool,
    policy_node_limit: u64,
    policy_deadline_milliseconds: i64,
}

impl<'a> Search<'a> {
    pub fn create_search(
        evaluation_network: &'a Network,
        position_table: &'a mut TranspositionTable,
        search_control: &'a SearchControl,
        search_options: SearchOptions,
    ) -> EngineResult<Self> {
        if !(1..=16).contains(&search_options.variation_count) || (search_options.puct_simulation_count > 4096)
            || !(0..=3).contains(&search_options.policy_verification_depth_plies)
            || !(0..=100).contains(&search_options.aggression_margin_centipawns)
            || !(0..=100).contains(&search_options.policy_uncertainty_percent) {
            return Err(EngineError::create_invalid_input("Failed to create a search because its options are invalid."));
        }

        return Ok(Self {
            network: evaluation_network,
            transposition_table: position_table,
            control: search_control,
            options: search_options,
            limits: SearchLimits::default(),
            started_at: Instant::now(),
            node_count: 0,
            selective_depth_plies: 0,
            was_interrupted: false,
            variations: [[Move::default(); MAXIMUM_PLY_COUNT]; MAXIMUM_PLY_COUNT],
            variation_lengths: [0; MAXIMUM_PLY_COUNT],
            killers: [[Move::default(); 2]; MAXIMUM_PLY_COUNT],
            quiet_history: [[[0; BOARD_SQUARE_COUNT]; BOARD_SQUARE_COUNT]; 2],
            capture_history: [[[0; 7]; BOARD_SQUARE_COUNT]; 12],
            continuation_history: vec![0; CONTINUATION_CONTEXT_COUNT * CONTINUATION_CONTEXT_COUNT],
            correction_history: [[0; CORRECTION_ENTRY_COUNT]; 2],
            previous_contexts: [None; MAXIMUM_PLY_COUNT],
            previous_moves: [Move::default(); MAXIMUM_PLY_COUNT],
            extension_counts: [0; MAXIMUM_PLY_COUNT],
            excluded_moves: [Move::default(); MAXIMUM_PLY_COUNT],
            is_null_verification: false,
            policy_node_limit: 0,
            policy_deadline_milliseconds: 0,
        });
    }

    fn should_stop_search(&mut self) -> bool {
        let deadline_milliseconds = self.control.hard_deadline_milliseconds.load(Ordering::Relaxed);
        let has_exhausted_nodes = (self.limits.node_count != 0) && (self.node_count >= self.limits.node_count);
        let has_expired_clock = (deadline_milliseconds != 0) && (get_monotonic_milliseconds() >= deadline_milliseconds);
        let has_exhausted_policy = (self.policy_node_limit != 0) && (self.node_count >= self.policy_node_limit);
        let has_expired_policy = (self.policy_deadline_milliseconds != 0) && (get_monotonic_milliseconds() >= self.policy_deadline_milliseconds);
        self.was_interrupted = self.was_interrupted || self.control.should_stop.load(Ordering::Relaxed)
            || has_exhausted_nodes || has_expired_clock || has_exhausted_policy || has_expired_policy;

        return self.was_interrupted;
    }

    fn get_static_score(
        &self,
        position: &Position,
        accumulator: &Accumulator,
    ) -> i32 {
        let raw_score = self.network.evaluate_position(
            position,
            accumulator,
        );
        let side = get_color_index(position.get_side_to_move());
        let index = position.get_pawn_key() as usize % CORRECTION_ENTRY_COUNT;

        let correction = if self.options.is_selective_search_enabled {
            self.correction_history[side][index] as i32 / 16
        } else {
            0
        };

        return (raw_score + correction).clamp(
            -20000,
            20000,
        );
    }

    fn update_history(
        history: &mut i16,
        bonus: i32,
    ) {
        let bounded_bonus = bonus.clamp(
            -2000,
            2000,
        );
        let previous = *history as i32;
        *history = (previous + bounded_bonus - previous * bounded_bonus.abs() / HISTORY_LIMIT) as i16;
    }

    fn update_variation(
        &mut self,
        ply: usize,
        chess_move: Move,
    ) {
        self.variations[ply][ply] = chess_move;
        self.variation_lengths[ply] = self.variation_lengths[ply + 1];

        for index in (ply + 1)..self.variation_lengths[ply] {
            self.variations[ply][index] = self.variations[ply + 1][index];
        }
    }

    fn order_moves(
        &self,
        position: &Position,
        accumulator: &Accumulator,
        moves: &mut MoveList,
        transposition_move: Move,
        ply: usize,
    ) {
        let mut scores = [0; MAXIMUM_MOVE_COUNT];
        let policy_scores = self.network.evaluate_policies(
            position,
            accumulator,
            moves,
        );

        for (index, &chess_move) in moves.moves[..moves.count].iter().enumerate() {
            let mut score = (policy_scores[index] * 100.0) as i32;
            let piece_index = get_piece_index(position.get_piece(chess_move.get_origin()));
            let destination = chess_move.get_destination() as usize;

            if chess_move == transposition_move {
                score += 10000000;
            } else if chess_move.is_tactical() {
                let victim_score = if chess_move.has_flag(MoveFlag::EN_PASSANT) {
                    PIECE_SCORES[1]
                } else {
                    get_piece_score(position.get_piece(chess_move.get_destination()))
                };

                let attacker_score = get_piece_score(position.get_piece(chess_move.get_origin()));
                let exchange_score = calculate_exchange_score(
                    position,
                    chess_move,
                );
                let capture_priority = if exchange_score >= 0 {
                    1000000
                } else {
                    -1000000
                };
                score += capture_priority + (victim_score * 16) - attacker_score + PIECE_SCORES[chess_move.get_promotion() as usize];
                let captured_type = position.get_piece(chess_move.get_destination()).unsigned_abs() as usize;
                score += self.capture_history[piece_index][destination][captured_type] as i32;
            } else if chess_move == self.killers[ply][0] {
                score += 900000;
            } else if chess_move == self.killers[ply][1] {
                score += 800000;
            } else {
                score += self.quiet_history[get_color_index(position.get_side_to_move())][chess_move.get_origin() as usize][chess_move.get_destination() as usize];

                if ply > 0 {
                    if let Some(previous_context) = self.previous_contexts[ply - 1] {
                        let context = piece_index * BOARD_SQUARE_COUNT + destination;
                        score += self.continuation_history[previous_context * CONTINUATION_CONTEXT_COUNT + context] as i32;
                    }
                }
            }

            scores[index] = score;
        }

        for index in 1..moves.count {
            let chess_move = moves.moves[index];
            let score = scores[index];
            let mut insertion_index = index;

            while (insertion_index > 0) && (scores[insertion_index - 1] < score) {
                moves.moves[insertion_index] = moves.moves[insertion_index - 1];
                scores[insertion_index] = scores[insertion_index - 1];
                insertion_index -= 1;
            }

            moves.moves[insertion_index] = chess_move;
            scores[insertion_index] = score;
        }
    }

    fn search_quiescence(
        &mut self,
        position: &mut Position,
        accumulator: &Accumulator,
        mut alpha: i32,
        beta: i32,
        ply: usize,
    ) -> EngineResult<i32> {
        self.variation_lengths[ply] = ply;

        if self.should_stop_search() {
            return Ok(0);
        }

        self.node_count += 1;
        self.selective_depth_plies = self.selective_depth_plies.max(ply);
        let is_in_check = position.is_in_check(position.get_side_to_move());
        let mut moves = position.generate_legal_moves(!is_in_check)?;

        if (moves.count == 0) && (is_in_check || !position.has_legal_move()?) {
            return Ok(if is_in_check {
                -MATE_SCORE + ply as i32
            } else {
                0
            });
        }

        if position.is_rule_draw() {
            return Ok(0);
        }

        let static_score = self.get_static_score(
            position,
            accumulator,
        );

        if ply >= (MAXIMUM_PLY_COUNT - 2) {
            return Ok(static_score);
        }

        if !is_in_check {
            if static_score >= beta {
                return Ok(static_score);
            }

            alpha = alpha.max(static_score);
        }

        self.order_moves(
            position,
            accumulator,
            &mut moves,
            Move::default(),
            ply,
        );

        for &chess_move in &moves.moves[..moves.count] {
            self.previous_moves[ply] = chess_move;
            self.previous_contexts[ply] = Some(get_piece_index(position.get_piece(chess_move.get_origin())) * BOARD_SQUARE_COUNT + chess_move.get_destination() as usize);
            let undo = position.make_move(chess_move)?;
            let next_accumulator = self.network.update_accumulator(
                accumulator,
                &undo,
            );
            let searched_score = self.search_quiescence(
                position,
                &next_accumulator,
                -beta,
                -alpha,
                ply + 1,
            );
            position.unmake_move(&undo);
            let score = -searched_score?;

            if self.was_interrupted {
                return Ok(0);
            }

            if score >= beta {
                return Ok(score);
            }

            if score > alpha {
                alpha = score;
                self.update_variation(
                    ply,
                    chess_move,
                );
            }
        }

        return Ok(alpha);
    }

    fn search_position(
        &mut self,
        position: &mut Position,
        accumulator: &Accumulator,
        depth_plies: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
    ) -> EngineResult<i32> {
        if depth_plies <= 0 {
            return self.search_quiescence(
                position,
                accumulator,
                alpha,
                beta,
                ply,
            );
        }

        self.variation_lengths[ply] = ply;

        if self.should_stop_search() {
            return Ok(0);
        }

        self.node_count += 1;
        self.selective_depth_plies = self.selective_depth_plies.max(ply);
        let is_in_check = position.is_in_check(position.get_side_to_move());
        let mut moves = position.generate_legal_moves(false)?;

        if moves.count == 0 {
            return Ok(if is_in_check {
                -MATE_SCORE + ply as i32
            } else {
                0
            });
        }

        if position.is_rule_draw() {
            return Ok(0);
        }

        if ply >= (MAXIMUM_PLY_COUNT - 2) {
            return Ok(self.network.evaluate_position(
                position,
                accumulator,
            ));
        }

        alpha = alpha.max(-MATE_SCORE + ply as i32);
        beta = beta.min(MATE_SCORE - ply as i32 - 1);

        if alpha >= beta {
            return Ok(alpha);
        }

        let original_alpha = alpha;
        let search_key = position.get_search_key();
        let cached = self.transposition_table.probe_position(search_key);
        let transposition_move = cached.map_or(
            Move::default(),
            |entry| entry.best_move,
        );
        let is_principal_node = (beta - alpha) > 1;
        let has_excluded_move = self.excluded_moves[ply].encoding != 0;

        if let Some(entry) = cached {
            if (entry.depth_plies as i32 >= depth_plies) && !is_principal_node && !has_excluded_move {
                let score = TranspositionTable::decode_score(
                    entry.score as i32,
                    ply,
                );

                if (entry.bound == BoundType::EXACT) || ((entry.bound == BoundType::LOWER) && (score >= beta)) || ((entry.bound == BoundType::UPPER) && (score <= alpha)) {
                    return Ok(score);
                }
            }
        }

        let static_score = self.get_static_score(
            position,
            accumulator,
        );

        if self.options.is_selective_search_enabled && !is_principal_node && !is_in_check && !has_excluded_move
            && (depth_plies <= 3) && (beta.abs() < MATE_THRESHOLD)
            && (static_score >= beta + REVERSE_FUTILITY_MARGIN_CENTIPAWNS_PER_PLY * depth_plies)
            && position.has_non_pawn_material() && (position.get_halfmove_count() < 80) {
            return Ok(static_score);
        }

        if self.options.is_selective_search_enabled && !self.is_null_verification && !has_excluded_move
            && !is_principal_node && !is_in_check && (depth_plies >= 4) && (static_score >= beta)
            && (beta.abs() < MATE_THRESHOLD) && position.has_non_pawn_material()
            && (position.get_halfmove_count() < 80) && (ply > 0) && (self.previous_moves[ply - 1].encoding != 0) {
            let undo = position.make_null_move()?;
            self.previous_moves[ply] = Move::default();
            self.previous_contexts[ply] = None;
            self.extension_counts[ply + 1] = self.extension_counts[ply];
            let null_score = self.search_position(
                position,
                accumulator,
                depth_plies - 3,
                -beta,
                -beta + 1,
                ply + 1,
            );
            position.unmake_move(&undo);
            let null_score = -null_score?;

            if !self.was_interrupted && (null_score >= beta) && (null_score.abs() < MATE_THRESHOLD) {
                self.is_null_verification = true;
                let verification = self.search_position(
                    position,
                    accumulator,
                    depth_plies - 3,
                    beta - 1,
                    beta,
                    ply,
                );
                self.is_null_verification = false;
                let verified_score = verification?;

                if !self.was_interrupted && (verified_score >= beta) {
                    return Ok(verified_score);
                }
            }
        }

        let mut singular_extension = 0;

        if self.options.is_selective_search_enabled && !has_excluded_move && !is_in_check
            && (depth_plies >= 6) && (self.extension_counts[ply] < MAXIMUM_EXTENSION_COUNT) {
            if let Some(entry) = cached {
                let cached_score = TranspositionTable::decode_score(
                    entry.score as i32,
                    ply,
                );

                if (entry.depth_plies as i32 >= depth_plies - 3) && (entry.bound == BoundType::LOWER)
                    && (cached_score.abs() < MATE_THRESHOLD) && (entry.best_move.encoding != 0) {
                    let singular_beta = cached_score - depth_plies * SINGULAR_MARGIN_CENTIPAWNS_PER_PLY;
                    self.excluded_moves[ply] = entry.best_move;
                    let alternative_score = self.search_position(
                        position,
                        accumulator,
                        depth_plies / 2,
                        singular_beta - 1,
                        singular_beta,
                        ply,
                    );
                    self.excluded_moves[ply] = Move::default();

                    if alternative_score? < singular_beta {
                        singular_extension = 1;
                    }
                }
            }
        }

        self.order_moves(
            position,
            accumulator,
            &mut moves,
            transposition_move,
            ply,
        );
        let mut best_move = Move::default();
        let mut best_score = -INFINITY_SCORE;
        let mut searched_move_count = 0;

        for (index, &chess_move) in moves.moves[..moves.count].iter().enumerate() {
            if chess_move == self.excluded_moves[ply] {
                continue;
            }

            searched_move_count += 1;
            let piece_index = get_piece_index(position.get_piece(chess_move.get_origin()));
            let captured_type = position.get_piece(chess_move.get_destination()).unsigned_abs() as usize;
            self.previous_moves[ply] = chess_move;
            self.previous_contexts[ply] = Some(piece_index * BOARD_SQUARE_COUNT + chess_move.get_destination() as usize);
            let undo = position.make_move(chess_move)?;
            let next_accumulator = self.network.update_accumulator(
                accumulator,
                &undo,
            );
            let does_give_check = position.is_in_check(position.get_side_to_move());
            let should_reduce = self.options.is_selective_search_enabled && (depth_plies >= 3) && (index >= 4)
                && !is_principal_node && !is_in_check && !chess_move.is_tactical() && !does_give_check;
            let extension = if self.options.is_selective_search_enabled && (self.extension_counts[ply] < MAXIMUM_EXTENSION_COUNT)
                && ((is_in_check && (depth_plies <= 2)) || ((chess_move == transposition_move) && (singular_extension > 0))) {
                1
            } else {
                0
            };
            self.extension_counts[ply + 1] = self.extension_counts[ply] + extension as usize;
            let child_depth_plies = depth_plies - 1 + extension;
            let searched_score = (|| -> EngineResult<i32> {
                if searched_move_count == 1 {
                    return Ok(-self.search_position(
                        position,
                        &next_accumulator,
                        child_depth_plies,
                        -beta,
                        -alpha,
                        ply + 1,
                    )?);
                }

                let mut score = -self.search_position(
                    position,
                    &next_accumulator,
                    child_depth_plies - i32::from(should_reduce),
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                )?;

                if should_reduce && (score > alpha) && !self.was_interrupted {
                    score = -self.search_position(
                        position,
                        &next_accumulator,
                        child_depth_plies,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                    )?;
                }

                if (score > alpha) && (score < beta) && !self.was_interrupted {
                    score = -self.search_position(
                        position,
                        &next_accumulator,
                        child_depth_plies,
                        -beta,
                        -alpha,
                        ply + 1,
                    )?;
                }

                return Ok(score);
            })();
            position.unmake_move(&undo);
            let score = searched_score?;

            if self.was_interrupted {
                return Ok(0);
            }

            if score > best_score {
                best_score = score;
                best_move = chess_move;
            }

            if score > alpha {
                alpha = score;
                self.update_variation(
                    ply,
                    chess_move,
                );
            }

            if alpha >= beta {
                if !chess_move.is_tactical() {
                    self.killers[ply][1] = self.killers[ply][0];
                    self.killers[ply][0] = chess_move;
                    let side_index = get_color_index(position.get_side_to_move());
                    let origin = chess_move.get_origin() as usize;
                    let destination = chess_move.get_destination() as usize;
                    let history_score = &mut self.quiet_history[side_index][origin][destination];
                    let bonus = (depth_plies * depth_plies * 16).min(1000);
                    *history_score += bonus - ((*history_score * bonus) / HISTORY_LIMIT);

                    if ply > 0 {
                        if let Some(previous_context) = self.previous_contexts[ply - 1] {
                            let context = piece_index * BOARD_SQUARE_COUNT + destination;
                            Self::update_history(
                                &mut self.continuation_history[previous_context * CONTINUATION_CONTEXT_COUNT + context],
                                bonus,
                            );
                        }
                    }
                } else {
                    Self::update_history(
                        &mut self.capture_history[piece_index][chess_move.get_destination() as usize][captured_type],
                        depth_plies * depth_plies * 16,
                    );
                }

                break;
            }
        }

        if searched_move_count == 0 {
            return Ok(alpha);
        }

        let bound = if best_score >= beta {
            BoundType::LOWER
        } else if best_score > original_alpha {
            BoundType::EXACT
        } else {
            BoundType::UPPER
        };

        if !has_excluded_move {
            self.transposition_table.store_position(
                search_key,
                best_move,
                best_score,
                depth_plies,
                bound,
                ply,
            );

            if self.options.is_selective_search_enabled && !is_in_check && !best_move.is_tactical()
                && (bound == BoundType::EXACT) && (best_score.abs() < MATE_THRESHOLD) {
                let side = get_color_index(position.get_side_to_move());
                let correction_index = position.get_pawn_key() as usize % CORRECTION_ENTRY_COUNT;
                let correction = (best_score - static_score).clamp(
                    -200,
                    200,
                );
                Self::update_history(
                    &mut self.correction_history[side][correction_index],
                    correction * depth_plies.min(8),
                );
            }
        }

        return Ok(best_score);
    }

    fn explore_policy_tree(
        &mut self,
        root: &Position,
        root_moves: &MoveList,
    ) -> EngineResult<[f64; MAXIMUM_MOVE_COUNT]> {
        let mut root_scores = [0.0; MAXIMUM_MOVE_COUNT];

        if (self.options.puct_simulation_count == 0) || (root_moves.count == 0) {
            return Ok(root_scores);
        }

        let mut nodes = Vec::with_capacity(MAXIMUM_PUCT_NODE_COUNT);
        let mut edges: Vec<PolicyEdge> = Vec::with_capacity(MAXIMUM_PUCT_EDGE_COUNT);
        nodes.push(PolicyNode::default());
        let root_accumulator = self.network.build_accumulator(root);
        let hard_deadline = self.control.hard_deadline_milliseconds.load(Ordering::Relaxed);
        let now_milliseconds = get_monotonic_milliseconds();
        self.policy_node_limit = if self.limits.node_count == 0 {
            (self.options.puct_simulation_count as u64 * POLICY_NODE_ALLOWANCE_PER_SIMULATION).max(1)
        } else {
            (self.limits.node_count / POLICY_BUDGET_DIVISOR).max(1)
        };
        self.policy_deadline_milliseconds = if hard_deadline > now_milliseconds {
            now_milliseconds + ((hard_deadline - now_milliseconds) / POLICY_BUDGET_DIVISOR as i64).max(1)
        } else {
            0
        };

        for _ in 0..self.options.puct_simulation_count {
            if self.should_stop_search() {
                break;
            }

            self.node_count += 1;
            let mut position = root.clone();
            let mut accumulator = root_accumulator;
            let mut path_edges = [0; MAXIMUM_PLY_COUNT];
            let mut path_nodes = [0; MAXIMUM_PLY_COUNT];
            let mut path_count = 0;
            let mut node_index = 0;

            while nodes[node_index].is_expanded && !nodes[node_index].is_terminal && (path_count < (MAXIMUM_PLY_COUNT - 2)) {
                let node = &nodes[node_index];
                let mut selected_index = None;
                let mut best_priority = f64::NEG_INFINITY;
                let exploration_scale = PUCT_EXPLORATION_FACTOR * ((node.visit_count + 1) as f64).sqrt();

                let first_play_value = if node.visit_count > 0 {
                    node.value_sum / node.visit_count as f64
                } else {
                    0.0
                } - 0.2;

                for (index, edge) in edges.iter().enumerate().skip(node.first_edge).take(node.edge_count) {
                    let mean_value = if edge.visit_count > 0 {
                        edge.value_sum / edge.visit_count as f64
                    } else {
                        first_play_value
                    };

                    let empirical_variance = if edge.visit_count > 1 {
                        (edge.value_squared_sum / edge.visit_count as f64 - mean_value * mean_value).max(0.0)
                    } else {
                        1.0
                    };
                    let uncertainty = ((empirical_variance + POLICY_VARIANCE_FLOOR) * ((node.visit_count + 2) as f64).ln()
                        / (edge.visit_count + 1) as f64).sqrt();
                    let priority = mean_value + ((exploration_scale * edge.prior) / (edge.visit_count + 1) as f64)
                        + uncertainty * self.options.policy_uncertainty_percent as f64 / 100.0;

                    if priority > best_priority {
                        best_priority = priority;
                        selected_index = Some(index);
                    }
                }

                let Some(selected_index) = selected_index else {
                    break;
                };
                let edge = &mut edges[selected_index];

                if edge.child_index.is_none() {
                    if nodes.len() >= MAXIMUM_PUCT_NODE_COUNT {
                        break;
                    }

                    edge.child_index = Some(nodes.len());
                    nodes.push(PolicyNode::default());
                }

                path_edges[path_count] = selected_index;
                path_nodes[path_count] = node_index;
                path_count += 1;
                self.previous_moves[path_count - 1] = edge.chess_move;
                self.previous_contexts[path_count - 1] = Some(get_piece_index(position.get_piece(edge.chess_move.get_origin())) * BOARD_SQUARE_COUNT + edge.chess_move.get_destination() as usize);
                self.extension_counts[path_count] = 0;
                let undo = position.make_move(edge.chess_move)?;
                accumulator = self.network.update_accumulator(
                    &accumulator,
                    &undo,
                );
                node_index = edge.child_index.expect("Failed to resolve a policy tree child.");
            }

            let leaf = &mut nodes[node_index];
            let mut position_value = self.network.evaluate_expected_outcome(
                &position,
                &accumulator,
            );

            if !leaf.is_expanded {
                let legal_moves = if node_index == 0 {
                    root_moves.clone()
                } else {
                    position.generate_legal_moves(false)?
                };

                if (legal_moves.count == 0) || position.is_rule_draw() {
                    leaf.is_terminal = true;

                    leaf.terminal_value = if (legal_moves.count == 0) && position.is_in_check(position.get_side_to_move()) {
                        -1.0
                    } else {
                        0.0
                    };

                    leaf.is_expanded = true;
                } else if (edges.len() + legal_moves.count) <= MAXIMUM_PUCT_EDGE_COUNT {
                    let mut logits = self.network.evaluate_policies(
                        &position,
                        &accumulator,
                        &legal_moves,
                    );
                    let mut maximum_logit = f64::NEG_INFINITY;

                    for &logit in &logits[..legal_moves.count] {
                        maximum_logit = maximum_logit.max(logit);
                    }

                    let mut policy_sum = 0.0;

                    for logit in &mut logits[..legal_moves.count] {
                        *logit = (*logit - maximum_logit).exp();
                        policy_sum += *logit;
                    }

                    leaf.first_edge = edges.len();
                    leaf.edge_count = legal_moves.count;
                    leaf.is_expanded = true;

                    for (index, &chess_move) in legal_moves.moves[..legal_moves.count].iter().enumerate() {
                        edges.push(PolicyEdge {
                            chess_move,
                            child_index: None,
                            visit_count: 0,
                            prior: logits[index] / policy_sum,
                            value_sum: 0.0,
                            value_squared_sum: 0.0,
                        });
                    }
                }
            }

            if leaf.is_terminal {
                position_value = leaf.terminal_value;
            } else if self.options.policy_verification_depth_plies > 0 {
                let verified_score = self.search_position(
                    &mut position,
                    &accumulator,
                    self.options.policy_verification_depth_plies,
                    -INFINITY_SCORE,
                    INFINITY_SCORE,
                    path_count,
                )?;

                if !self.was_interrupted {
                    position_value = (verified_score as f64 / 400.0).tanh();
                }
            }

            leaf.visit_count += 1;
            leaf.value_sum += position_value;

            for path_index in (0..path_count).rev() {
                position_value = -position_value;
                let edge = &mut edges[path_edges[path_index]];
                let node = &mut nodes[path_nodes[path_index]];
                edge.visit_count += 1;
                edge.value_sum += position_value;
                edge.value_squared_sum += position_value * position_value;
                node.visit_count += 1;
                node.value_sum += position_value;
            }
        }

        self.policy_node_limit = 0;
        self.policy_deadline_milliseconds = 0;
        self.was_interrupted = false;
        self.should_stop_search();

        if nodes[0].edge_count != 0 {
            for index in 0..root_moves.count {
                let edge = &edges[nodes[0].first_edge + index];
                root_scores[index] = edge.visit_count as f64 + edge.prior;
            }
        }

        return Ok(root_scores);
    }

    pub fn run_search(
        &mut self,
        mut position: Position,
        requested_limits: &SearchLimits,
        observer: &mut dyn FnMut(&SearchReport),
    ) -> EngineResult<SearchResult> {
        if (requested_limits.depth_plies < 1) || (requested_limits.depth_plies >= (MAXIMUM_PLY_COUNT as i32 - 1)) {
            return Err(EngineError::create_invalid_input("Failed to search because the depth is invalid."));
        }

        if position.get_history_count() > (MAXIMUM_HISTORY_COUNT - MAXIMUM_PLY_COUNT) {
            return Err(EngineError::create_invalid_input("Failed to search because its move history exceeds the supported capacity."));
        }

        self.limits = requested_limits.clone();
        self.node_count = 0;
        self.selective_depth_plies = 0;
        self.was_interrupted = false;
        self.previous_contexts.fill(None);
        self.previous_moves.fill(Move::default());
        self.extension_counts.fill(0);
        self.excluded_moves.fill(Move::default());
        self.is_null_verification = false;
        self.started_at = Instant::now();
        self.transposition_table.start_generation();
        let mut root_moves = position.generate_legal_moves(false)?;

        if !self.limits.root_moves.is_empty() {
            let mut filtered_moves = MoveList::default();

            for &chess_move in &root_moves.moves[..root_moves.count] {
                if self.limits.root_moves.contains(&chess_move) {
                    filtered_moves.add_move(chess_move)?;
                }
            }

            root_moves = filtered_moves;
        }

        let mut search_result = SearchResult::default();

        if root_moves.count == 0 {
            search_result.score = if position.is_in_check(position.get_side_to_move()) {
                -MATE_SCORE
            } else {
                0
            };

            return Ok(search_result);
        }

        search_result.best_move = root_moves.moves[0];
        search_result.principal_variation.push(search_result.best_move);

        if position.is_rule_draw() {
            return Ok(search_result);
        }

        let root_policy_scores = self.explore_policy_tree(
            &position,
            &root_moves,
        )?;
        self.previous_contexts.fill(None);
        self.extension_counts.fill(0);
        let mut root_indices: Vec<usize> = (0..root_moves.count).collect();
        root_indices.sort_by(|
            &first,
            &second,
        | root_policy_scores[second].total_cmp(&root_policy_scores[first]));
        let mut root_variations: Vec<RootVariation> = root_indices.iter().map(|&index| RootVariation {
            chess_move: root_moves.moves[index],
            score: -INFINITY_SCORE,
            principal_variation: Vec::new(),
        }).collect();
        let accumulator = self.network.build_accumulator(&position);
        let mut previous_best_change_count = 0;

        for depth_plies in 1..=self.limits.depth_plies {
            let mut completed_variations = Vec::with_capacity(root_variations.len());

            for candidate in &root_variations {
                if self.should_stop_search() {
                    break;
                }

                let mut aspiration_window_score = if depth_plies >= 4 {
                    ASPIRATION_WINDOW_SCORE
                } else {
                    INFINITY_SCORE
                };

                let mut alpha = if candidate.score == -INFINITY_SCORE {
                    -INFINITY_SCORE
                } else {
                    (candidate.score - aspiration_window_score).max(-INFINITY_SCORE)
                };

                let mut beta = if candidate.score == -INFINITY_SCORE {
                    INFINITY_SCORE
                } else {
                    (candidate.score + aspiration_window_score).min(INFINITY_SCORE)
                };

                let undo = position.make_move(candidate.chess_move)?;
                self.previous_moves[0] = candidate.chess_move;
                self.previous_contexts[0] = Some(get_piece_index(undo.changes[0].before) * BOARD_SQUARE_COUNT + candidate.chess_move.get_destination() as usize);
                self.extension_counts[1] = 0;
                let next_accumulator = self.network.update_accumulator(
                    &accumulator,
                    &undo,
                );
                let searched_score = (|| -> EngineResult<i32> {
                    if (self.options.variation_count == 1) && !completed_variations.is_empty() {
                        let best_root_score = completed_variations.iter().map(|variation: &RootVariation| variation.score)
                            .max().unwrap_or(-INFINITY_SCORE);
                        let threshold = best_root_score - self.options.aggression_margin_centipawns - 1;

                        if threshold > -INFINITY_SCORE {
                            let bounded_score = -self.search_position(
                                &mut position,
                                &next_accumulator,
                                depth_plies - 1,
                                -threshold - 1,
                                -threshold,
                                1,
                            )?;

                            if self.was_interrupted || (bounded_score <= threshold) {
                                return Ok(bounded_score);
                            }
                        }
                    }

                    loop {
                        let score = -self.search_position(
                            &mut position,
                            &next_accumulator,
                            depth_plies - 1,
                            -beta,
                            -alpha,
                            1,
                        )?;

                        if self.was_interrupted || ((score > alpha) && (score < beta)) {
                            return Ok(score);
                        }

                        aspiration_window_score = (aspiration_window_score * 2).min(INFINITY_SCORE);
                        alpha = (score - aspiration_window_score).max(-INFINITY_SCORE);
                        beta = (score + aspiration_window_score).min(INFINITY_SCORE);
                    }
                })();
                position.unmake_move(&undo);
                let score = searched_score?;

                if self.was_interrupted {
                    break;
                }

                let mut principal_variation = vec![candidate.chess_move,];
                principal_variation.extend_from_slice(&self.variations[1][1..self.variation_lengths[1]]);
                completed_variations.push(RootVariation {
                    chess_move: candidate.chess_move,
                    score,
                    principal_variation,
                });
            }

            if self.was_interrupted || (completed_variations.len() != root_variations.len()) {
                break;
            }

            completed_variations.sort_by_key(|variation| Reverse(variation.score));

            if completed_variations[0].score.abs() < MATE_THRESHOLD {
                let score_floor = completed_variations[0].score - self.options.aggression_margin_centipawns;
                let mut preferred_index = 0;
                let mut preferred_priority = i32::MIN;

                for (index, candidate) in completed_variations.iter().enumerate() {
                    if candidate.score < score_floor {
                        break;
                    }

                    let priority = calculate_attack_priority(
                        &position,
                        candidate.chess_move,
                    );

                    if priority > preferred_priority {
                        preferred_priority = priority;
                        preferred_index = index;
                    }
                }

                let preferred = completed_variations.remove(preferred_index);
                completed_variations.insert(
                    0,
                    preferred,
                );
            }

            root_variations = completed_variations;
            previous_best_change_count += i32::from(search_result.best_move != root_variations[0].chess_move);
            search_result.best_move = root_variations[0].chess_move;
            search_result.score = root_variations[0].score;
            search_result.completed_depth_plies = depth_plies;
            search_result.principal_variation = root_variations[0].principal_variation.clone();
            let elapsed_milliseconds = self.started_at.elapsed().as_millis() as i64;
            let reported_count = self.options.variation_count.min(root_variations.len());

            for (index, variation) in root_variations.iter().take(reported_count).enumerate() {
                observer(&SearchReport {
                    depth_plies,
                    selective_depth_plies: self.selective_depth_plies,
                    score: variation.score,
                    variation_index: index + 1,
                    hashfull_per_mille: self.transposition_table.get_hashfull_per_mille(),
                    node_count: self.node_count,
                    elapsed_milliseconds,
                    principal_variation: variation.principal_variation.clone(),
                });
            }

            let soft_deadline = self.control.soft_deadline_milliseconds.load(Ordering::Relaxed);
            let is_stable = (depth_plies >= 4) && (previous_best_change_count <= 1);

            if (soft_deadline != 0) && is_stable && (get_monotonic_milliseconds() >= soft_deadline) {
                break;
            }

            if !self.limits.is_infinite && !self.control.is_pondering.load(Ordering::SeqCst) && (search_result.score.abs() >= MATE_THRESHOLD) {
                break;
            }
        }

        search_result.node_count = self.node_count;

        return Ok(search_result);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::{
        calculate_time_budget,
        SearchControl,
        SearchLimits,
        TimeBudget,
    };

    #[test]
    fn test_time_budgets_remain_ordered_and_preserve_available_clock() {
        for remaining_milliseconds in [0, 1, 20, 100, 1000, 86400000,] {
            for increment_milliseconds in [0, 50, 1000, 60000,] {
                for moves_to_go in [0, 1, 30, 1000,] {
                    let budget = calculate_time_budget(
                        remaining_milliseconds,
                        increment_milliseconds,
                        moves_to_go,
                        20,
                    );
                    assert!(budget.optimum_milliseconds >= 1);
                    assert!(budget.maximum_milliseconds >= budget.optimum_milliseconds);
                    assert!(budget.maximum_milliseconds <= (remaining_milliseconds - 20).max(1));
                }
            }
        }
    }

    #[test]
    fn test_search_control_resets_stop_and_defers_ponder_and_infinite_clocks() {
        let control = SearchControl::default();

        for (is_pondering, is_infinite) in [(true, false,), (false, true,),] {
            control.should_stop.store(
                true,
                Ordering::SeqCst,
            );
            control.start_search(&SearchLimits {
                is_pondering,
                is_infinite,
                time: TimeBudget {
                    optimum_milliseconds: 100,
                    maximum_milliseconds: 200,
                },
                ..SearchLimits::default()
            });
            assert!(!control.should_stop.load(Ordering::SeqCst));
            assert_eq!(
                control.is_pondering.load(Ordering::SeqCst),
                is_pondering,
            );
            assert_eq!(
                control.hard_deadline_milliseconds.load(Ordering::SeqCst),
                0,
            );
        }
    }

    #[test]
    fn test_ponderhit_activates_ordered_deadlines() {
        let control = SearchControl::default();
        control.start_search(&SearchLimits {
            is_pondering: true,
            ..SearchLimits::default()
        });
        control.activate_clock(TimeBudget {
            optimum_milliseconds: 100,
            maximum_milliseconds: 200,
        });
        assert!(!control.is_pondering.load(Ordering::SeqCst));
        assert!(control.soft_deadline_milliseconds.load(Ordering::SeqCst) > 0);
        assert_eq!(
            control.hard_deadline_milliseconds.load(Ordering::SeqCst) - control.soft_deadline_milliseconds.load(Ordering::SeqCst),
            100,
        );
    }
}
