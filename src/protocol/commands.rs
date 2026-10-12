use crate::chess::position::Position;
use crate::chess::types::{
    Color,
    EngineError,
    EngineResult,
    MAXIMUM_PLY_COUNT,
};
use crate::engine::search::{
    calculate_time_budget,
    SearchLimits,
    TimeBudget,
};

const MAXIMUM_CLOCK_MILLISECONDS: i64 = 604800000;

pub(super) fn parse_integer(
    text: &str,
    minimum: i64,
    maximum: i64,
) -> EngineResult<i64> {
    let integer = text.parse::<i64>().map_err(|_| EngineError::create_invalid_input("Failed to parse a command integer because it is outside its supported range."))?;

    if text.starts_with('+') || (integer < minimum) || (integer > maximum) {
        return Err(EngineError::create_invalid_input("Failed to parse a command integer because it is outside its supported range."));
    }

    return Ok(integer);
}

pub(super) fn parse_search_limits(
    tokens: &[&str],
    position: &mut Position,
    move_overhead_milliseconds: i64,
) -> EngineResult<SearchLimits> {
    let mut limits = SearchLimits::default();
    let mut remaining_milliseconds = 0;
    let mut increment_milliseconds = 0;
    let mut move_time_milliseconds = 0;
    let mut moves_to_go = 0;
    let mut has_clock = false;
    let mut has_move_time = false;
    let side = position.get_side_to_move();
    let mut index = 1;

    while index < tokens.len() {
        let name = tokens[index];
        index += 1;

        if name == "infinite" {
            limits.is_infinite = true;

            continue;
        }

        if name == "ponder" {
            limits.is_pondering = true;

            continue;
        }

        if name == "searchmoves" {
            while index < tokens.len() {
                let move_name = tokens[index];
                let symbols = move_name.as_bytes();
                let is_move_token = ((symbols.len() == 4) || (symbols.len() == 5))
                    && (b'a'..=b'h').contains(&symbols[0]) && (b'1'..=b'8').contains(&symbols[1])
                    && (b'a'..=b'h').contains(&symbols[2]) && (b'1'..=b'8').contains(&symbols[3]);

                if !is_move_token {
                    break;
                }

                limits.root_moves.push(position.find_legal_move(move_name)?);
                index += 1;
            }

            if limits.root_moves.is_empty() {
                return Err(EngineError::create_invalid_input("Failed to start a search because searchmoves requires legal moves."));
            }

            continue;
        }

        if index >= tokens.len() {
            return Err(EngineError::create_invalid_input("Failed to start a search because an argument value is missing."));
        }

        let argument = tokens[index];
        index += 1;

        match name {
            "depth" => limits.depth_plies = parse_integer(
                argument,
                1,
                MAXIMUM_PLY_COUNT as i64 - 2,
            )? as i32,
            "nodes" => limits.node_count = parse_integer(
                argument,
                1,
                1000000000000,
            )? as u64,
            "movetime" => {
                move_time_milliseconds = parse_integer(
                    argument,
                    1,
                    MAXIMUM_CLOCK_MILLISECONDS,
                )?;
                has_move_time = true;
            }
            "movestogo" => moves_to_go = parse_integer(
                argument,
                1,
                1000,
            )? as i32,
            "wtime" | "btime" | "winc" | "binc" => {
                let milliseconds = parse_integer(
                    argument,
                    0,
                    MAXIMUM_CLOCK_MILLISECONDS,
                )?;
                let is_own_clock = ((side == Color::WHITE) && name.starts_with('w')) || ((side == Color::BLACK) && name.starts_with('b'));

                if is_own_clock && name.ends_with("time") {
                    remaining_milliseconds = milliseconds;
                    has_clock = true;
                } else if is_own_clock {
                    increment_milliseconds = milliseconds;
                }
            }
            "mate" => limits.depth_plies = parse_integer(
                argument,
                1,
                (MAXIMUM_PLY_COUNT as i64 - 2) / 2,
            )? as i32 * 2,
            _ => return Err(EngineError::create_invalid_input("Failed to start a search because an argument is unsupported.")),
        }
    }

    if has_move_time {
        let available_milliseconds = (move_time_milliseconds - move_overhead_milliseconds).max(1);
        limits.time = TimeBudget {
            optimum_milliseconds: available_milliseconds,
            maximum_milliseconds: available_milliseconds,
        };
    } else if has_clock {
        limits.time = calculate_time_budget(
            remaining_milliseconds,
            increment_milliseconds,
            moves_to_go,
            move_overhead_milliseconds,
        );
    }

    return Ok(limits);
}
