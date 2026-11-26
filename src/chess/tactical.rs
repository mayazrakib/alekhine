use crate::chess::position::Position;
use crate::chess::types::{
    BOARD_SQUARE_COUNT,
    Color,
    get_piece_score,
    Move,
    MoveFlag,
    PIECE_SCORES,
};

const EXCHANGE_KING_SCORE: i32 = 20000;

pub(crate) fn can_attack_square(
    board: &[i32; BOARD_SQUARE_COUNT],
    origin: usize,
    destination: usize,
) -> bool {
    let piece = board[origin];
    let file_distance = (destination % 8) as i32 - (origin % 8) as i32;
    let rank_distance = (destination / 8) as i32 - (origin / 8) as i32;
    let absolute_file = file_distance.abs();
    let absolute_rank = rank_distance.abs();

    match piece.abs() {
        1 => return (absolute_file == 1) && (rank_distance == piece.signum()),
        2 => return (absolute_file * absolute_rank) == 2,
        6 => return absolute_file.max(absolute_rank) == 1,
        3 => {
            if absolute_file != absolute_rank {
                return false;
            }
        }
        4 => {
            if (file_distance != 0) && (rank_distance != 0) {
                return false;
            }
        }
        5 => {
            if (absolute_file != absolute_rank) && (file_distance != 0) && (rank_distance != 0) {
                return false;
            }
        }
        _ => return false,
    }

    if origin == destination {
        return false;
    }

    let file_step = file_distance.signum();
    let rank_step = rank_distance.signum();
    let mut file = (origin % 8) as i32 + file_step;
    let mut rank = (origin / 8) as i32 + rank_step;

    while (file != (destination % 8) as i32) || (rank != (destination / 8) as i32) {
        if board[(rank * 8 + file) as usize] != 0 {
            return false;
        }

        file += file_step;
        rank += rank_step;
    }

    return true;
}

fn has_attacked_king(
    board: &[i32; BOARD_SQUARE_COUNT],
    sign: i32,
) -> bool {
    let Some(king_square) = board.iter().position(|&piece| piece == sign * 6) else {
        return true;
    };

    return board.iter().enumerate().any(|(square, &piece)| {
        return (piece.signum() == -sign) && can_attack_square(
            board,
            square,
            king_square,
        );
    });
}

pub fn calculate_exchange_score(
    position: &Position,
    chess_move: Move,
) -> i32 {
    if chess_move.has_flag(MoveFlag::CASTLE) {
        return 0;
    }

    let mut board = [0; BOARD_SQUARE_COUNT];

    for (square, piece) in board.iter_mut().enumerate() {
        *piece = position.get_piece(square as i32);
    }

    let origin = chess_move.get_origin() as usize;
    let destination = chess_move.get_destination() as usize;
    let moving_piece = board[origin];
    let mut gains = [0; 32];
    gains[0] = get_piece_score(board[destination]);
    board[origin] = 0;

    if chess_move.has_flag(MoveFlag::EN_PASSANT) {
        gains[0] = PIECE_SCORES[1];
        board[(destination as i32 - moving_piece.signum() * 8) as usize] = 0;
    }

    board[destination] = moving_piece;
    let promotion = chess_move.get_promotion() as usize;

    if promotion != 0 {
        gains[0] += PIECE_SCORES[promotion] - PIECE_SCORES[1];
        board[destination] = moving_piece.signum() * promotion as i32;
    }

    let mut sign = -moving_piece.signum();
    let mut depth = 0;

    while depth < gains.len() - 1 {
        let mut attacker = None;
        let mut attacker_score = i32::MAX;

        for square in 0..BOARD_SQUARE_COUNT {
            let piece = board[square];

            if (square == destination) || (piece.signum() != sign) || !can_attack_square(
                &board,
                square,
                destination,
            ) {
                continue;
            }

            let piece_score = if piece.abs() == 6 {
                EXCHANGE_KING_SCORE
            } else {
                get_piece_score(piece)
            };

            if piece_score >= attacker_score {
                continue;
            }

            let mut next_board = board;
            next_board[destination] = piece;
            next_board[square] = 0;

            if !has_attacked_king(
                &next_board,
                sign,
            ) {
                attacker = Some(square);
                attacker_score = piece_score;
            }
        }

        let Some(attacker) = attacker else {
            break;
        };
        depth += 1;
        gains[depth] = get_piece_score(board[destination]) - gains[depth - 1];
        board[destination] = board[attacker];
        board[attacker] = 0;

        if (board[destination].abs() == 1) && ((destination / 8 == 0) || (destination / 8 == 7)) {
            board[destination] = sign * 5;
            gains[depth] += PIECE_SCORES[5] - PIECE_SCORES[1];
        }

        sign = -sign;
    }

    while depth > 0 {
        gains[depth - 1] = -(-gains[depth - 1]).max(gains[depth]);
        depth -= 1;
    }

    return gains[0];
}

pub fn calculate_attack_priority(
    position: &Position,
    chess_move: Move,
) -> i32 {
    let mut child = position.clone();
    let Ok(undo) = child.make_move(chess_move) else {
        return 0;
    };
    let attacker = position.get_side_to_move();
    let defender = child.get_side_to_move();
    let mut priority = i32::from(child.is_in_check(defender)) * 1000;
    let defender_sign = if defender == Color::WHITE {
        1
    } else {
        -1
    };
    let king_square = child.get_pieces(defender_sign * 6).trailing_zeros() as i32;

    for square in 0..BOARD_SQUARE_COUNT as i32 {
        let file_distance = ((square % 8) - (king_square % 8)).abs();
        let rank_distance = ((square / 8) - (king_square / 8)).abs();

        if (file_distance.max(rank_distance) <= 1) && child.is_square_attacked(
            square,
            attacker,
        ) {
            priority += 40;
        }
    }

    if chess_move.is_tactical() {
        priority += 20;
    }

    child.unmake_move(&undo);

    return priority;
}

#[cfg(test)]
mod tests {
    use crate::chess::position::Position;
    use crate::chess::types::{
        BOARD_SQUARE_COUNT,
        Color,
        START_FEN,
    };
    use super::{
        calculate_attack_priority,
        can_attack_square,
    };

    #[test]
    fn test_geometric_attacks_match_position_attack_maps() {
        for fen in [START_FEN, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",] {
            let position = Position::parse_fen(fen).unwrap();
            let mut board = [0; BOARD_SQUARE_COUNT];

            for (square, piece) in board.iter_mut().enumerate() {
                *piece = position.get_piece(square as i32);
            }

            for (color, sign) in [(Color::WHITE, 1,), (Color::BLACK, -1,),] {
                for target in 0..BOARD_SQUARE_COUNT {
                    let has_attack = board.iter().enumerate().any(|(origin, &piece)| {
                        return (piece.signum() == sign) && can_attack_square(
                            &board,
                            origin,
                            target,
                        );
                    });
                    assert_eq!(
                        has_attack,
                        position.is_square_attacked(
                            target as i32,
                            color,
                        ),
                    );
                }
            }
        }
    }

    #[test]
    fn test_slider_attacks_stop_at_blockers_and_pawns_use_color_direction() {
        let mut board = [0; BOARD_SQUARE_COUNT];
        board[0] = 4;
        board[16] = 1;
        board[27] = -1;
        assert!(can_attack_square(
            &board,
            0,
            16,
        ));
        assert!(!can_attack_square(
            &board,
            0,
            24,
        ));
        assert!(can_attack_square(
            &board,
            16,
            25,
        ));
        assert!(!can_attack_square(
            &board,
            16,
            9,
        ));
        assert!(can_attack_square(
            &board,
            27,
            18,
        ));
        assert!(!can_attack_square(
            &board,
            27,
            34,
        ));
    }

    #[test]
    fn test_checking_move_priority_preserves_original_position() {
        let mut position = Position::parse_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap();
        let original_fen = position.get_fen();
        let original_key = position.get_key();
        let checking_move = position.find_legal_move("g6g7").unwrap();
        let quiet_move = position.find_legal_move("g6d3").unwrap();
        assert!(calculate_attack_priority(
            &position,
            checking_move,
        ) > calculate_attack_priority(
            &position,
            quiet_move,
        ));
        assert_eq!(
            position.get_fen(),
            original_fen,
        );
        assert_eq!(
            position.get_key(),
            original_key,
        );
    }
}
