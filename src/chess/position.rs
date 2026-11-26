use crate::chess::types::{
    Bitboard,
    BLACK_KINGSIDE,
    BLACK_QUEENSIDE,
    BOARD_SQUARE_COUNT,
    Color,
    EngineError,
    EngineResult,
    get_color_index,
    get_opponent,
    get_piece_color,
    get_piece_index,
    get_piece_score,
    get_piece_type,
    get_square_bit,
    get_square_name,
    MAXIMUM_HISTORY_COUNT,
    MAXIMUM_PLY_COUNT,
    Move,
    MoveFlag,
    MoveList,
    NO_SQUARE,
    parse_square,
    PieceType,
    WHITE_KINGSIDE,
    WHITE_QUEENSIDE,
};

const KING_DIRECTIONS: [[i32; 2]; 8] = [[1, 0,], [-1, 0,], [0, 1,], [0, -1,], [1, 1,], [1, -1,], [-1, 1,], [-1, -1,],];
const KNIGHT_DIRECTIONS: [[i32; 2]; 8] = [[1, 2,], [2, 1,], [-1, 2,], [-2, 1,], [1, -2,], [2, -1,], [-1, -2,], [-2, -1,],];
const PIECE_NAMES: &[u8] = b"PNBRQKpnbrqk";
const KNIGHT_ATTACKS: [Bitboard; BOARD_SQUARE_COUNT] = build_attack_masks(&KNIGHT_DIRECTIONS);
const KING_ATTACKS: [Bitboard; BOARD_SQUARE_COUNT] = build_attack_masks(&KING_DIRECTIONS);
const ZOBRIST_KEYS: ZobristKeys = build_zobrist_keys();
const DARK_SQUARES: Bitboard = 0xaa55aa55aa55aa55;
const BACK_RANKS: Bitboard = 0xff000000000000ff;
const CONTEXT_HASH_MULTIPLIER: u64 = 0x9e3779b97f4a7c15;
const NULL_CONTEXT_HASH_MULTIPLIER: u64 = 0xd6e8feb86659fd93;

struct ZobristKeys {
    pieces: [[u64; BOARD_SQUARE_COUNT]; 12],
    castling: [u64; 16],
    en_passant: [u64; 16],
    side: u64,
}

const fn generate_random_key(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut key = *seed;
    key = (key ^ (key >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    key = (key ^ (key >> 27)).wrapping_mul(0x94d049bb133111eb);

    return key ^ (key >> 31);
}

const fn build_zobrist_keys() -> ZobristKeys {
    let mut keys = ZobristKeys {
        pieces: [[0; BOARD_SQUARE_COUNT]; 12],
        castling: [0; 16],
        en_passant: [0; 16],
        side: 0,
    };
    let mut seed = 0x616c656b68696e65;
    let mut piece_index = 0;

    while piece_index < 12 {
        let mut square = 0;

        while square < BOARD_SQUARE_COUNT {
            keys.pieces[piece_index][square] = generate_random_key(&mut seed);
            square += 1;
        }

        piece_index += 1;
    }

    let mut index = 0;

    while index < 16 {
        keys.castling[index] = generate_random_key(&mut seed);
        index += 1;
    }

    index = 0;

    while index < 16 {
        keys.en_passant[index] = generate_random_key(&mut seed);
        index += 1;
    }

    keys.side = generate_random_key(&mut seed);

    return keys;
}

const fn build_attack_masks(directions: &[[i32; 2]; 8]) -> [Bitboard; BOARD_SQUARE_COUNT] {
    let mut attacks = [0; BOARD_SQUARE_COUNT];
    let mut square = 0;

    while square < BOARD_SQUARE_COUNT {
        let mut index = 0;

        while index < directions.len() {
            let file = (square as i32 % 8) + directions[index][0];
            let rank = (square as i32 / 8) + directions[index][1];

            if (file >= 0) && (file < 8) && (rank >= 0) && (rank < 8) {
                attacks[square] |= 1 << ((rank * 8) + file);
            }

            index += 1;
        }

        square += 1;
    }

    return attacks;
}

fn parse_counter(
    text: &str,
    minimum: i32,
) -> EngineResult<i32> {
    let counter = text.parse::<i32>().map_err(|_| EngineError::create_invalid_input("Failed to parse a FEN move counter."))?;

    if text.starts_with('+') || (counter < minimum) || (counter > 1000000) {
        return Err(EngineError::create_invalid_input("Failed to parse a FEN move counter."));
    }

    return Ok(counter);
}

fn add_pawn_move(
    moves: &mut MoveList,
    origin: i32,
    destination: i32,
    flag: MoveFlag,
) -> EngineResult<()> {
    if ((destination / 8) == 0) || ((destination / 8) == 7) {
        for promotion in [PieceType::QUEEN, PieceType::ROOK, PieceType::BISHOP, PieceType::KNIGHT,] {
            moves.add_move(Move::create_move(
                origin,
                destination,
                promotion,
                flag,
            ))?;
        }

        return Ok(());
    }

    moves.add_move(Move::create_move(
        origin,
        destination,
        PieceType::NONE,
        flag,
    ))?;

    return Ok(());
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
)]
pub struct PieceChange {
    pub square: i32,
    pub before: i32,
    pub after: i32,
}

#[derive(
    Clone,
    Debug,
)]
pub struct UndoState {
    pub changes: [PieceChange; 4],
    pub change_count: usize,
    castling_rights: i32,
    en_passant_square: i32,
    halfmove_count: i32,
    fullmove_number: i32,
    history_count: usize,
    key: u64,
    pawn_key: u64,
    null_depth: usize,
}

#[derive(Clone)]
pub struct Position {
    board: [i8; BOARD_SQUARE_COUNT],
    piece_bitboards: [Bitboard; 12],
    occupancies: [Bitboard; 2],
    history: [u64; MAXIMUM_HISTORY_COUNT],
    side_to_move: Color,
    castling_rights: i32,
    en_passant_square: i32,
    halfmove_count: i32,
    fullmove_number: i32,
    history_count: usize,
    material_score: i32,
    key: u64,
    pawn_key: u64,
    null_depth: usize,
}

impl Position {
    pub fn parse_fen(fen: &str) -> EngineResult<Self> {
        if fen.len() > 256 {
            return Err(EngineError::create_invalid_input("Failed to parse FEN because its length exceeds 256 bytes."));
        }

        let fields: Vec<&str> = fen.split_whitespace().collect();

        if fields.len() != 6 {
            return Err(EngineError::create_invalid_input("Failed to parse FEN because exactly six fields are required."));
        }

        let mut position = Self {
            board: [0; BOARD_SQUARE_COUNT],
            piece_bitboards: [0; 12],
            occupancies: [0; 2],
            history: [0; MAXIMUM_HISTORY_COUNT],
            side_to_move: Color::WHITE,
            castling_rights: 0,
            en_passant_square: NO_SQUARE,
            halfmove_count: 0,
            fullmove_number: 1,
            history_count: 0,
            material_score: 0,
            key: 0,
            pawn_key: 0,
            null_depth: 0,
        };
        let mut rank = 7;
        let mut file = 0;

        for symbol in fields[0].bytes() {
            if symbol == b'/' {
                if (file != 8) || (rank == 0) {
                    return Err(EngineError::create_invalid_input("Failed to parse FEN because its ranks are malformed."));
                }

                rank -= 1;
                file = 0;

                continue;
            }

            if (b'1'..=b'8').contains(&symbol) {
                file += (symbol - b'0') as i32;
            } else {
                let piece_index = PIECE_NAMES.iter().position(|&piece_symbol| piece_symbol == symbol)
                    .ok_or_else(|| EngineError::create_invalid_input("Failed to parse FEN because its placement is malformed."))?;

                if file >= 8 {
                    return Err(EngineError::create_invalid_input("Failed to parse FEN because its placement is malformed."));
                }

                let piece = if piece_index < 6 {
                    piece_index as i32 + 1
                } else {
                    -(piece_index as i32 - 5)
                };

                position.replace_piece(
                    (rank * 8) + file,
                    piece,
                );
                file += 1;
            }

            if file > 8 {
                return Err(EngineError::create_invalid_input("Failed to parse FEN because a rank exceeds eight squares."));
            }
        }

        if (rank != 0) || (file != 8) || ((fields[1] != "w") && (fields[1] != "b")) {
            return Err(EngineError::create_invalid_input("Failed to parse FEN because its placement or side is invalid."));
        }

        position.side_to_move = if fields[1] == "w" {
            Color::WHITE
        } else {
            Color::BLACK
        };

        position.halfmove_count = parse_counter(
            fields[4],
            0,
        )?;
        position.fullmove_number = parse_counter(
            fields[5],
            1,
        )?;

        if fields[2] != "-" {
            for right in fields[2].bytes() {
                let right_index = b"KQkq".iter().position(|&symbol| symbol == right)
                    .ok_or_else(|| EngineError::create_invalid_input("Failed to parse FEN because its castling rights are invalid."))?;

                if (position.castling_rights & (1 << right_index)) != 0 {
                    return Err(EngineError::create_invalid_input("Failed to parse FEN because its castling rights are invalid."));
                }

                position.castling_rights |= 1 << right_index;
            }
        }

        for color_index in 0..2 {
            let offset = color_index * 56;

            let sign = if color_index == 0 {
                1
            } else {
                -1
            };

            let rights = position.castling_rights >> (color_index * 2);

            if ((rights & 3) != 0) && (position.board[offset + 4] != (sign * 6)) {
                return Err(EngineError::create_invalid_input("Failed to parse FEN because a castling king is absent."));
            }

            if (((rights & 1) != 0) && (position.board[offset + 7] != (sign * 4))) || (((rights & 2) != 0) && (position.board[offset] != (sign * 4))) {
                return Err(EngineError::create_invalid_input("Failed to parse FEN because a castling rook is absent."));
            }
        }

        if fields[3] != "-" {
            let square = parse_square(fields[3])?;

            let expected_rank = if position.side_to_move == Color::WHITE {
                5
            } else {
                2
            };

            let captured_square = square + if position.side_to_move == Color::WHITE {
                -8
            } else {
                8
            };
            let origin_square = square + if position.side_to_move == Color::WHITE {
                8
            } else {
                -8
            };

            let expected_pawn = if position.side_to_move == Color::WHITE {
                -1
            } else {
                1
            };

            if (square / 8) != expected_rank {
                return Err(EngineError::create_invalid_input("Failed to parse FEN because its en passant state is invalid."));
            }

            let has_valid_en_passant = (position.get_piece(square) == 0)
                && (position.get_piece(captured_square) == expected_pawn)
                && (position.get_piece(origin_square) == 0)
                && (position.halfmove_count == 0);

            if !has_valid_en_passant {
                return Err(EngineError::create_invalid_input("Failed to parse FEN because its en passant state is invalid."));
            }

            position.en_passant_square = square;
        }

        position.key = position.compute_key();
        position.history[0] = position.key;
        position.history_count = 1;

        if !position.has_valid_state() || position.is_in_check(get_opponent(position.side_to_move)) {
            return Err(EngineError::create_invalid_input("Failed to parse FEN because the position is illegal."));
        }

        return Ok(position);
    }

    pub fn get_fen(&self) -> String {
        let mut fen = String::new();

        for rank in (0..8).rev() {
            let mut empty_count = 0;

            for file in 0..8 {
                let piece = self.get_piece((rank * 8) + file);

                if piece == 0 {
                    empty_count += 1;

                    continue;
                }

                if empty_count != 0 {
                    fen.push((b'0' + empty_count) as char);
                    empty_count = 0;
                }

                fen.push(PIECE_NAMES[get_piece_index(piece)] as char);
            }

            if empty_count != 0 {
                fen.push((b'0' + empty_count) as char);
            }

            if rank != 0 {
                fen.push('/');
            }
        }

        fen.push_str(if self.side_to_move == Color::WHITE {
            " w "
        } else {
            " b "
        });

        if self.castling_rights == 0 {
            fen.push('-');
        } else {
            for index in 0..4 {
                if (self.castling_rights & (1 << index)) != 0 {
                    fen.push(b"KQkq"[index] as char);
                }
            }
        }

        let en_passant = if self.en_passant_square == NO_SQUARE {
            "-".to_owned()
        } else {
            get_square_name(self.en_passant_square).expect("Failed to format the en passant square.")
        };

        fen.push_str(&format!(
            " {en_passant} {} {}",
            self.halfmove_count,
            self.fullmove_number,
        ));

        return fen;
    }

    pub fn get_piece(
        &self,
        square: i32,
    ) -> i32 {
        return self.board[square as usize] as i32;
    }

    pub fn get_side_to_move(&self) -> Color {
        return self.side_to_move;
    }

    pub fn get_pieces(
        &self,
        piece: i32,
    ) -> Bitboard {
        return self.piece_bitboards[get_piece_index(piece)];
    }

    pub fn get_occupancy(&self) -> Bitboard {
        return self.occupancies[0] | self.occupancies[1];
    }

    pub fn get_key(&self) -> u64 {
        return self.key;
    }

    pub fn get_pawn_key(&self) -> u64 {
        return self.pawn_key;
    }

    pub fn get_search_key(&self) -> u64 {
        let mut context = self.halfmove_count as u64;
        let first_index = self.history_count.saturating_sub(self.halfmove_count as usize + 1);

        for index in first_index..self.history_count {
            context = context.rotate_left(7) ^ self.history[index];
        }

        return self.key ^ context.wrapping_mul(CONTEXT_HASH_MULTIPLIER) ^ (self.null_depth as u64).wrapping_mul(NULL_CONTEXT_HASH_MULTIPLIER);
    }

    pub fn get_halfmove_count(&self) -> i32 {
        return self.halfmove_count;
    }

    pub fn get_material_score(&self) -> i32 {
        return self.material_score;
    }

    pub fn get_history_count(&self) -> usize {
        return self.history_count;
    }

    pub fn is_square_attacked(
        &self,
        square: i32,
        attacker: Color,
    ) -> bool {
        let sign = if attacker == Color::WHITE {
            1
        } else {
            -1
        };

        let file = square % 8;
        let rank = square / 8;
        let pawn_rank = rank + if attacker == Color::WHITE {
            -1
        } else {
            1
        };

        if (0..8).contains(&pawn_rank) {
            for file_delta in [-1, 1,] {
                let pawn_file = file + file_delta;

                if (0..8).contains(&pawn_file) && (self.get_piece((pawn_rank * 8) + pawn_file) == sign) {
                    return true;
                }
            }
        }

        if ((KNIGHT_ATTACKS[square as usize] & self.get_pieces(sign * 2)) != 0) || ((KING_ATTACKS[square as usize] & self.get_pieces(sign * 6)) != 0) {
            return true;
        }

        for (index, direction) in KING_DIRECTIONS.iter().enumerate() {
            let mut target_file = file + direction[0];
            let mut target_rank = rank + direction[1];

            while (0..8).contains(&target_file) && (0..8).contains(&target_rank) {
                let piece = self.get_piece((target_rank * 8) + target_file);

                if piece != 0 {
                    let piece_type = get_piece_type(piece);
                    let is_slider = (piece_type == PieceType::QUEEN) || if index < 4 {
                        piece_type == PieceType::ROOK
                    } else {
                        piece_type == PieceType::BISHOP
                    };

                    if (get_piece_color(piece) == attacker) && is_slider {
                        return true;
                    }

                    break;
                }

                target_file += direction[0];
                target_rank += direction[1];
            }
        }

        return false;
    }

    pub fn is_in_check(
        &self,
        color: Color,
    ) -> bool {
        let king_piece = if color == Color::WHITE {
            6
        } else {
            -6
        };

        let king = self.get_pieces(king_piece);

        if king == 0 {
            return true;
        }

        return self.is_square_attacked(
            king.trailing_zeros() as i32,
            get_opponent(color),
        );
    }

    pub fn is_repetition(
        &self,
        occurrence_count: usize,
    ) -> bool {
        if self.null_depth != 0 {
            return false;
        }

        let mut match_count = 0;
        let first_index = self.history_count.saturating_sub(self.halfmove_count as usize + 1);

        for index in (first_index..self.history_count).rev().step_by(2) {
            if self.history[index] == self.key {
                match_count += 1;

                if match_count >= occurrence_count {
                    return true;
                }
            }
        }

        return false;
    }

    pub fn has_insufficient_material(&self) -> bool {
        if (self.get_pieces(1) | self.get_pieces(-1) | self.get_pieces(4) | self.get_pieces(-4) | self.get_pieces(5) | self.get_pieces(-5)) != 0 {
            return false;
        }

        let bishops = self.get_pieces(3) | self.get_pieces(-3);
        let knights = self.get_pieces(2) | self.get_pieces(-2);

        if (bishops | knights).count_ones() <= 1 {
            return true;
        }

        if knights != 0 {
            return false;
        }

        return ((bishops & DARK_SQUARES) == 0) || ((bishops & !DARK_SQUARES) == 0);
    }

    pub fn is_rule_draw(&self) -> bool {
        return ((self.null_depth == 0) && ((self.halfmove_count >= 100) || self.is_repetition(3))) || self.has_insufficient_material();
    }

    fn replace_piece(
        &mut self,
        square: i32,
        piece: i32,
    ) {
        let square_index = square as usize;
        let previous_piece = self.get_piece(square);
        let bit = get_square_bit(square);

        if previous_piece != 0 {
            let piece_index = get_piece_index(previous_piece);
            self.piece_bitboards[piece_index] ^= bit;
            self.occupancies[get_color_index(get_piece_color(previous_piece))] ^= bit;
            self.key ^= ZOBRIST_KEYS.pieces[piece_index][square_index];
            self.material_score -= previous_piece.signum() * get_piece_score(previous_piece);

            if get_piece_type(previous_piece) == PieceType::PAWN {
                self.pawn_key ^= ZOBRIST_KEYS.pieces[piece_index][square_index];
            }
        }

        self.board[square_index] = piece as i8;

        if piece != 0 {
            let piece_index = get_piece_index(piece);
            self.piece_bitboards[piece_index] |= bit;
            self.occupancies[get_color_index(get_piece_color(piece))] |= bit;
            self.key ^= ZOBRIST_KEYS.pieces[piece_index][square_index];
            self.material_score += piece.signum() * get_piece_score(piece);

            if get_piece_type(piece) == PieceType::PAWN {
                self.pawn_key ^= ZOBRIST_KEYS.pieces[piece_index][square_index];
            }
        }
    }

    fn apply_change(
        &mut self,
        undo: &mut UndoState,
        square: i32,
        piece: i32,
    ) {
        undo.changes[undo.change_count] = PieceChange {
            square,
            before: self.get_piece(square),
            after: piece,
        };
        undo.change_count += 1;
        self.replace_piece(
            square,
            piece,
        );
    }

    fn has_legal_en_passant(&self) -> bool {
        if self.en_passant_square == NO_SQUARE {
            return false;
        }

        let sign = if self.side_to_move == Color::WHITE {
            1
        } else {
            -1
        };

        let captured_square = self.en_passant_square - (sign * 8);

        for delta in [-1, 1,] {
            let origin = captured_square + delta;

            if !(0..BOARD_SQUARE_COUNT as i32).contains(&origin) || ((origin / 8) != (captured_square / 8)) || (self.get_piece(origin) != sign) {
                continue;
            }

            let mut candidate = self.clone();
            candidate.replace_piece(
                origin,
                0,
            );
            candidate.replace_piece(
                captured_square,
                0,
            );
            candidate.replace_piece(
                self.en_passant_square,
                sign,
            );

            if !candidate.is_in_check(self.side_to_move) {
                return true;
            }
        }

        return false;
    }

    fn compute_key(&self) -> u64 {
        let mut computed_key = ZOBRIST_KEYS.castling[self.castling_rights as usize];

        for square in 0..BOARD_SQUARE_COUNT as i32 {
            let piece = self.get_piece(square);

            if piece != 0 {
                computed_key ^= ZOBRIST_KEYS.pieces[get_piece_index(piece)][square as usize];
            }
        }

        if self.side_to_move == Color::BLACK {
            computed_key ^= ZOBRIST_KEYS.side;
        }

        if self.has_legal_en_passant() {
            computed_key ^= ZOBRIST_KEYS.en_passant[(self.en_passant_square % 8) as usize];
        }

        return computed_key;
    }

    pub fn make_move(
        &mut self,
        chess_move: Move,
    ) -> EngineResult<UndoState> {
        if self.history_count >= MAXIMUM_HISTORY_COUNT {
            return Err(EngineError::create_invalid_input("Failed to make a move because the position history is full."));
        }

        let mut undo = UndoState {
            changes: [PieceChange::default(); 4],
            change_count: 0,
            castling_rights: self.castling_rights,
            en_passant_square: self.en_passant_square,
            halfmove_count: self.halfmove_count,
            fullmove_number: self.fullmove_number,
            history_count: self.history_count,
            key: self.key,
            pawn_key: self.pawn_key,
            null_depth: self.null_depth,
        };
        let origin = chess_move.get_origin();
        let destination = chess_move.get_destination();
        let piece = self.get_piece(origin);

        let sign = if self.side_to_move == Color::WHITE {
            1
        } else {
            -1
        };

        if self.has_legal_en_passant() {
            self.key ^= ZOBRIST_KEYS.en_passant[(self.en_passant_square % 8) as usize];
        }

        self.key ^= ZOBRIST_KEYS.castling[self.castling_rights as usize];
        self.apply_change(
            &mut undo,
            origin,
            0,
        );
        self.apply_change(
            &mut undo,
            destination,

            if chess_move.get_promotion() == PieceType::NONE {
                piece
            } else {
                sign * chess_move.get_promotion() as i32
            },
        );

        if chess_move.has_flag(MoveFlag::EN_PASSANT) {
            self.apply_change(
                &mut undo,
                destination - (sign * 8),
                0,
            );
        }

        if chess_move.has_flag(MoveFlag::CASTLE) {
            let rook_origin = if destination > origin {
                origin + 3
            } else {
                origin - 4
            };

            let rook_destination = (origin + destination) / 2;
            self.apply_change(
                &mut undo,
                rook_origin,
                0,
            );
            self.apply_change(
                &mut undo,
                rook_destination,
                sign * 4,
            );
        }

        if get_piece_type(piece) == PieceType::KING {
            self.castling_rights &= !(if self.side_to_move == Color::WHITE {
                3
            } else {
                12
            });
        }

        for square in [origin, destination,] {
            match square {
                0 => self.castling_rights &= !WHITE_QUEENSIDE,
                7 => self.castling_rights &= !WHITE_KINGSIDE,
                56 => self.castling_rights &= !BLACK_QUEENSIDE,
                63 => self.castling_rights &= !BLACK_KINGSIDE,
                _ => (),
            }
        }

        self.en_passant_square = if chess_move.has_flag(MoveFlag::DOUBLE_PUSH) {
            (origin + destination) / 2
        } else {
            NO_SQUARE
        };

        self.halfmove_count = if (get_piece_type(piece) == PieceType::PAWN) || chess_move.has_flag(MoveFlag::CAPTURE) {
            0
        } else {
            self.halfmove_count + 1
        };

        self.fullmove_number += i32::from(self.side_to_move == Color::BLACK);
        self.side_to_move = get_opponent(self.side_to_move);
        self.key ^= ZOBRIST_KEYS.side;
        self.key ^= ZOBRIST_KEYS.castling[self.castling_rights as usize];

        if self.has_legal_en_passant() {
            self.key ^= ZOBRIST_KEYS.en_passant[(self.en_passant_square % 8) as usize];
        }

        self.history[self.history_count] = self.key;
        self.history_count += 1;

        return Ok(undo);
    }

    pub fn unmake_move(
        &mut self,
        undo: &UndoState,
    ) {
        for change in undo.changes[..undo.change_count].iter().rev() {
            self.replace_piece(
                change.square,
                change.before,
            );
        }

        self.side_to_move = get_opponent(self.side_to_move);
        self.castling_rights = undo.castling_rights;
        self.en_passant_square = undo.en_passant_square;
        self.halfmove_count = undo.halfmove_count;
        self.fullmove_number = undo.fullmove_number;
        self.history_count = undo.history_count;
        self.key = undo.key;
        self.pawn_key = undo.pawn_key;
        self.null_depth = undo.null_depth;
    }

    pub fn make_null_move(&mut self) -> EngineResult<UndoState> {
        if self.is_in_check(self.side_to_move) || (self.history_count >= MAXIMUM_HISTORY_COUNT) {
            return Err(EngineError::create_invalid_input("Failed to make a null move while in check or with a full history."));
        }

        let undo = UndoState {
            changes: [PieceChange::default(); 4],
            change_count: 0,
            castling_rights: self.castling_rights,
            en_passant_square: self.en_passant_square,
            halfmove_count: self.halfmove_count,
            fullmove_number: self.fullmove_number,
            history_count: self.history_count,
            key: self.key,
            pawn_key: self.pawn_key,
            null_depth: self.null_depth,
        };

        if self.has_legal_en_passant() {
            self.key ^= ZOBRIST_KEYS.en_passant[(self.en_passant_square % 8) as usize];
        }

        self.en_passant_square = NO_SQUARE;
        self.halfmove_count += 1;
        self.fullmove_number += i32::from(self.side_to_move == Color::BLACK);
        self.side_to_move = get_opponent(self.side_to_move);
        self.key ^= ZOBRIST_KEYS.side;
        self.null_depth += 1;
        self.history[self.history_count] = self.key;
        self.history_count += 1;

        return Ok(undo);
    }

    pub fn has_non_pawn_material(&self) -> bool {
        let sign = if self.side_to_move == Color::WHITE {
            1
        } else {
            -1
        };

        return (2..=5).any(|piece| self.get_pieces(sign * piece) != 0);
    }

    fn generate_pseudo_legal_moves(
        &self,
        should_generate_tactical_only: bool,
    ) -> EngineResult<MoveList> {
        let mut moves = MoveList::default();

        let sign = if self.side_to_move == Color::WHITE {
            1
        } else {
            -1
        };

        let mut pieces = self.occupancies[get_color_index(self.side_to_move)];

        while pieces != 0 {
            let origin = pieces.trailing_zeros() as i32;
            pieces &= pieces - 1;
            let piece_type = get_piece_type(self.get_piece(origin));
            let file = origin % 8;
            let rank = origin / 8;

            if piece_type == PieceType::PAWN {
                let destination = origin + (sign * 8);

                if (0..BOARD_SQUARE_COUNT as i32).contains(&destination) && (self.get_piece(destination) == 0) {
                    if !should_generate_tactical_only || ((destination / 8) == 0) || ((destination / 8) == 7) {
                        add_pawn_move(
                            &mut moves,
                            origin,
                            destination,
                            MoveFlag::QUIET,
                        )?;
                    }

                    let start_rank = if self.side_to_move == Color::WHITE {
                        1
                    } else {
                        6
                    };

                    if !should_generate_tactical_only && (rank == start_rank) && (self.get_piece(destination + (sign * 8)) == 0) {
                        add_pawn_move(
                            &mut moves,
                            origin,
                            destination + (sign * 8),
                            MoveFlag::DOUBLE_PUSH,
                        )?;
                    }
                }

                for delta in [-1, 1,] {
                    let target = destination + delta;

                    if !(0..8).contains(&(file + delta)) || !(0..BOARD_SQUARE_COUNT as i32).contains(&target) {
                        continue;
                    }

                    let victim = self.get_piece(target);

                    if (victim != 0) && (get_piece_color(victim) != self.side_to_move) && (get_piece_type(victim) != PieceType::KING) {
                        add_pawn_move(
                            &mut moves,
                            origin,
                            target,
                            MoveFlag::CAPTURE,
                        )?;
                    } else if target == self.en_passant_square {
                        add_pawn_move(
                            &mut moves,
                            origin,
                            target,
                            MoveFlag(MoveFlag::CAPTURE.0 | MoveFlag::EN_PASSANT.0),
                        )?;
                    }
                }

                continue;
            }

            if (piece_type == PieceType::KNIGHT) || (piece_type == PieceType::KING) {
                let mut targets = if piece_type == PieceType::KNIGHT {
                    KNIGHT_ATTACKS[origin as usize]
                } else {
                    KING_ATTACKS[origin as usize]
                };

                targets &= !self.occupancies[get_color_index(self.side_to_move)];

                while targets != 0 {
                    let target = targets.trailing_zeros() as i32;
                    targets &= targets - 1;
                    let victim = self.get_piece(target);

                    if (get_piece_type(victim) == PieceType::KING) || (should_generate_tactical_only && (victim == 0)) {
                        continue;
                    }

                    moves.add_move(Move::create_move(
                        origin,
                        target,
                        PieceType::NONE,

                        if victim == 0 {
                            MoveFlag::QUIET
                        } else {
                            MoveFlag::CAPTURE
                        },
                    ))?;
                }

                continue;
            }

            for (index, direction) in KING_DIRECTIONS.iter().enumerate() {
                if ((piece_type == PieceType::BISHOP) && (index < 4)) || ((piece_type == PieceType::ROOK) && (index >= 4)) {
                    continue;
                }

                let mut target_file = file + direction[0];
                let mut target_rank = rank + direction[1];

                while (0..8).contains(&target_file) && (0..8).contains(&target_rank) {
                    let target = (target_rank * 8) + target_file;
                    let victim = self.get_piece(target);

                    if (victim != 0) && ((get_piece_color(victim) == self.side_to_move) || (get_piece_type(victim) == PieceType::KING)) {
                        break;
                    }

                    if !should_generate_tactical_only || (victim != 0) {
                        moves.add_move(Move::create_move(
                            origin,
                            target,
                            PieceType::NONE,

                            if victim == 0 {
                                MoveFlag::QUIET
                            } else {
                                MoveFlag::CAPTURE
                            },
                        ))?;
                    }

                    if victim != 0 {
                        break;
                    }

                    target_file += direction[0];
                    target_rank += direction[1];
                }
            }
        }

        if !should_generate_tactical_only && !self.is_in_check(self.side_to_move) {
            let offset = if self.side_to_move == Color::WHITE {
                0
            } else {
                56
            };

            let rights = self.castling_rights >> (get_color_index(self.side_to_move) * 2);

            for is_kingside in [true, false,] {
                let right = if is_kingside {
                    1
                } else {
                    2
                };

                let step = if is_kingside {
                    1
                } else {
                    -1
                };

                let king_square = offset + 4;

                let rook_square = if is_kingside {
                    offset + 7
                } else {
                    offset
                };

                let mut is_clear = (rights & right) != 0;
                let mut target = king_square + step;

                while target != rook_square {
                    is_clear = is_clear && (self.get_piece(target) == 0);
                    target += step;
                }

                if is_clear && !self.is_square_attacked(
                    king_square + step,
                    get_opponent(self.side_to_move),
                ) && !self.is_square_attacked(
                    king_square + (step * 2),
                    get_opponent(self.side_to_move),
                ) {
                    moves.add_move(Move::create_move(
                        king_square,
                        king_square + (step * 2),
                        PieceType::NONE,
                        MoveFlag::CASTLE,
                    ))?;
                }
            }
        }

        return Ok(moves);
    }

    pub fn generate_legal_moves(
        &mut self,
        should_generate_tactical_only: bool,
    ) -> EngineResult<MoveList> {
        let candidates = self.generate_pseudo_legal_moves(should_generate_tactical_only)?;
        let mut legal_moves = MoveList::default();
        let moving_side = self.side_to_move;

        for &chess_move in &candidates.moves[..candidates.count] {
            let undo = self.make_move(chess_move)?;
            let is_legal = !self.is_in_check(moving_side);
            self.unmake_move(&undo);

            if is_legal {
                legal_moves.add_move(chess_move)?;
            }
        }

        return Ok(legal_moves);
    }

    pub fn has_legal_move(&mut self) -> EngineResult<bool> {
        let side = self.side_to_move;
        let moves = self.generate_pseudo_legal_moves(false)?;

        for &chess_move in &moves.moves[..moves.count] {
            let undo = self.make_move(chess_move)?;
            let is_legal = !self.is_in_check(side);
            self.unmake_move(&undo);

            if is_legal {
                return Ok(true);
            }
        }

        return Ok(false);
    }

    pub fn find_legal_move(
        &mut self,
        uci_move: &str,
    ) -> EngineResult<Move> {
        let moves = self.generate_legal_moves(false)?;

        for &chess_move in &moves.moves[..moves.count] {
            if chess_move.get_uci() == uci_move {
                return Ok(chess_move);
            }
        }

        return Err(EngineError::create_invalid_input("Failed to apply an illegal UCI move."));
    }

    pub fn has_valid_state(&self) -> bool {
        if (self.get_pieces(6).count_ones() != 1) || (self.get_pieces(-6).count_ones() != 1) || ((self.occupancies[0] & self.occupancies[1]) != 0) {
            return false;
        }

        if (self.occupancies[0].count_ones() > 16) || (self.occupancies[1].count_ones() > 16) || (self.get_pieces(1).count_ones() > 8) || (self.get_pieces(-1).count_ones() > 8) {
            return false;
        }

        if ((self.get_pieces(1) | self.get_pieces(-1)) & BACK_RANKS) != 0 {
            return false;
        }

        if (KING_ATTACKS[self.get_pieces(6).trailing_zeros() as usize] & self.get_pieces(-6)) != 0 {
            return false;
        }

        let mut expected_pieces = [0; 12];
        let mut expected_occupancies = [0; 2];
        let mut expected_pawn_key = 0;
        let mut expected_material = 0;

        for square in 0..BOARD_SQUARE_COUNT as i32 {
            let piece = self.get_piece(square);

            if piece == 0 {
                continue;
            }

            let piece_index = get_piece_index(piece);
            expected_pieces[piece_index] |= get_square_bit(square);
            expected_occupancies[get_color_index(get_piece_color(piece))] |= get_square_bit(square);
            expected_material += piece.signum() * get_piece_score(piece);

            if get_piece_type(piece) == PieceType::PAWN {
                expected_pawn_key ^= ZOBRIST_KEYS.pieces[piece_index][square as usize];
            }
        }

        return (expected_pieces == self.piece_bitboards) && (expected_occupancies == self.occupancies)
            && (expected_material == self.material_score) && (expected_pawn_key == self.pawn_key)
            && (self.compute_key() == self.key) && (self.history_count > 0)
            && (self.history[self.history_count - 1] == self.key);
    }
}

pub fn count_perft(
    position: &mut Position,
    depth_plies: i32,
) -> EngineResult<u64> {
    if (depth_plies < 0) || (depth_plies >= MAXIMUM_PLY_COUNT as i32) {
        return Err(EngineError::create_invalid_input("Failed to count perft because the depth is invalid."));
    }

    if depth_plies == 0 {
        return Ok(1);
    }

    let moves = position.generate_legal_moves(false)?;

    if depth_plies == 1 {
        return Ok(moves.count as u64);
    }

    let mut node_count = 0;

    for &chess_move in &moves.moves[..moves.count] {
        let undo = position.make_move(chess_move)?;
        let subtree = count_perft(
            position,
            depth_plies - 1,
        );
        position.unmake_move(&undo);
        node_count += subtree?;
    }

    return Ok(node_count);
}
