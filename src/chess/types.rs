use std::error::Error;
use std::fmt;
use std::io;

pub type Bitboard = u64;
pub const BOARD_SQUARE_COUNT: usize = 64;
pub const MAXIMUM_MOVE_COUNT: usize = 256;
pub const MAXIMUM_PLY_COUNT: usize = 128;
pub const MAXIMUM_HISTORY_COUNT: usize = 4096;
pub const MATE_SCORE: i32 = 30000;
pub const MATE_THRESHOLD: i32 = MATE_SCORE - MAXIMUM_PLY_COUNT as i32;
pub const INFINITY_SCORE: i32 = 32000;
pub const NO_SQUARE: i32 = -1;
pub const WHITE_KINGSIDE: i32 = 1;
pub const WHITE_QUEENSIDE: i32 = 2;
pub const BLACK_KINGSIDE: i32 = 4;
pub const BLACK_QUEENSIDE: i32 = 8;
pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
pub const PIECE_SCORES: [i32; 7] = [0, 100, 320, 330, 500, 900, 0,];

#[derive(Debug)]
#[allow(non_camel_case_types)]
pub enum EngineError {
    INVALID_INPUT(String),
    IO(io::Error),
}

impl EngineError {
    pub fn create_invalid_input(message: &str) -> Self {
        return Self::INVALID_INPUT(message.to_owned());
    }
}

impl fmt::Display for EngineError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::INVALID_INPUT(message) => return formatter.write_str(message),
            Self::IO(cause) => return write!(
                formatter,
                "Failed to complete engine I/O: {cause}.",
            ),
        }
    }
}

impl Error for EngineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IO(cause) => return Some(cause),
            Self::INVALID_INPUT(_) => return None,
        }
    }
}

impl From<io::Error> for EngineError {
    fn from(cause: io::Error) -> Self {
        return Self::IO(cause);
    }
}

pub type EngineResult<T> = Result<
    T,
    EngineError,
>;

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
)]
#[allow(non_camel_case_types)]
pub enum Color {
    WHITE,
    BLACK,
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum PieceType {
    NONE,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
)]
pub struct MoveFlag(pub u32);

impl MoveFlag {
    pub const QUIET: Self = Self(0);
    pub const CAPTURE: Self = Self(1 << 15);
    pub const EN_PASSANT: Self = Self(1 << 16);
    pub const CASTLE: Self = Self(1 << 17);
    pub const DOUBLE_PUSH: Self = Self(1 << 18);
}

pub fn get_opponent(color: Color) -> Color {
    return if color == Color::WHITE {
        Color::BLACK
    } else {
        Color::WHITE
    };
}

pub fn get_color_index(color: Color) -> usize {
    return color as usize;
}

pub fn get_piece_type(piece: i32) -> PieceType {
    match piece.abs() {
        1 => return PieceType::PAWN,
        2 => return PieceType::KNIGHT,
        3 => return PieceType::BISHOP,
        4 => return PieceType::ROOK,
        5 => return PieceType::QUEEN,
        6 => return PieceType::KING,
        _ => return PieceType::NONE,
    }
}

pub fn get_piece_color(piece: i32) -> Color {
    return if piece > 0 {
        Color::WHITE
    } else {
        Color::BLACK
    };
}

pub fn get_piece_index(piece: i32) -> usize {
    return (if piece > 0 {
        0
    } else {
        6
    }) + piece.unsigned_abs() as usize - 1;
}

pub fn get_piece_score(piece: i32) -> i32 {
    return PIECE_SCORES[get_piece_type(piece) as usize];
}

pub fn get_square_bit(square: i32) -> Bitboard {
    return 1 << square;
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    PartialEq,
)]
pub struct Move {
    pub encoding: u32,
}

impl Move {
    pub fn create_move(
        origin: i32,
        destination: i32,
        promotion: PieceType,
        flag: MoveFlag,
    ) -> Self {
        return Self {
            encoding: (origin | (destination << 6) | ((promotion as i32) << 12)) as u32 | flag.0,
        };
    }

    pub fn get_origin(self) -> i32 {
        return (self.encoding & 63) as i32;
    }

    pub fn get_destination(self) -> i32 {
        return ((self.encoding >> 6) & 63) as i32;
    }

    pub fn get_promotion(self) -> PieceType {
        return get_piece_type(((self.encoding >> 12) & 7) as i32);
    }

    pub fn has_flag(
        self,
        flag: MoveFlag,
    ) -> bool {
        return (self.encoding & flag.0) != 0;
    }

    pub fn is_tactical(self) -> bool {
        return self.has_flag(MoveFlag::CAPTURE) || (self.get_promotion() != PieceType::NONE);
    }

    pub fn get_uci(self) -> String {
        if self.encoding == 0 {
            return "0000".to_owned();
        }

        let mut move_name = get_square_name(self.get_origin()).expect("Failed to format the move origin.")
            + &get_square_name(self.get_destination()).expect("Failed to format the move destination.");

        if self.get_promotion() != PieceType::NONE {
            move_name.push(b" pnbrqk"[self.get_promotion() as usize] as char);
        }

        return move_name;
    }
}

#[derive(Clone)]
pub struct MoveList {
    pub moves: [Move; MAXIMUM_MOVE_COUNT],
    pub count: usize,
}

impl Default for MoveList {
    fn default() -> Self {
        return Self {
            moves: [Move::default(); MAXIMUM_MOVE_COUNT],
            count: 0,
        };
    }
}

impl MoveList {
    pub fn add_move(
        &mut self,
        chess_move: Move,
    ) -> EngineResult<()> {
        if self.count >= MAXIMUM_MOVE_COUNT {
            return Err(EngineError::create_invalid_input("Failed to generate moves because the move capacity was exceeded."));
        }

        self.moves[self.count] = chess_move;
        self.count += 1;

        return Ok(());
    }
}

pub fn get_square_name(square: i32) -> EngineResult<String> {
    if !(0..BOARD_SQUARE_COUNT as i32).contains(&square) {
        return Err(EngineError::create_invalid_input("Failed to format an invalid square."));
    }

    let file = (b'a' + (square % 8) as u8) as char;
    let rank = (b'1' + (square / 8) as u8) as char;

    return Ok(format!("{file}{rank}"));
}

pub fn parse_square(square_name: &str) -> EngineResult<i32> {
    let symbols = square_name.as_bytes();

    if (symbols.len() != 2) || !(b'a'..=b'h').contains(&symbols[0]) || !(b'1'..=b'8').contains(&symbols[1]) {
        return Err(EngineError::create_invalid_input("Failed to parse an invalid square."));
    }

    return Ok((symbols[0] - b'a') as i32 + ((symbols[1] - b'1') as i32 * 8));
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::io;

    use super::{
        BOARD_SQUARE_COUNT,
        Color,
        EngineError,
        get_color_index,
        get_opponent,
        get_piece_color,
        get_piece_index,
        get_piece_score,
        get_piece_type,
        get_square_name,
        MAXIMUM_MOVE_COUNT,
        Move,
        MoveFlag,
        MoveList,
        parse_square,
        PIECE_SCORES,
        PieceType,
    };

    #[test]
    fn test_square_names_roundtrip_on_every_square() {
        for square in 0..BOARD_SQUARE_COUNT as i32 {
            let name = get_square_name(square).unwrap();
            assert_eq!(
                parse_square(&name).unwrap(),
                square,
            );
        }
    }

    #[test]
    fn test_square_parsing_rejects_invalid_coordinates() {
        for name in ["", "a0", "i1", "A1", "a9", "a11", "é1", " a1",] {
            assert!(parse_square(name).is_err());
        }

        assert!(get_square_name(-1).is_err());
        assert!(get_square_name(64).is_err());
    }

    #[test]
    fn test_move_encoding_preserves_coordinates_promotions_and_flags() {
        for promotion in [PieceType::NONE, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::QUEEN,] {
            for flag in [MoveFlag::QUIET, MoveFlag::CAPTURE, MoveFlag::EN_PASSANT, MoveFlag::CASTLE, MoveFlag::DOUBLE_PUSH,] {
                let chess_move = Move::create_move(
                    48,
                    56,
                    promotion,
                    flag,
                );
                assert_eq!(
                    chess_move.get_origin(),
                    48,
                );
                assert_eq!(
                    chess_move.get_destination(),
                    56,
                );
                assert_eq!(
                    chess_move.get_promotion(),
                    promotion,
                );
                assert_eq!(
                    chess_move.has_flag(flag),
                    flag != MoveFlag::QUIET,
                );
                assert_eq!(
                    chess_move.is_tactical(),
                    (flag == MoveFlag::CAPTURE) || (promotion != PieceType::NONE),
                );
            }
        }

        for (promotion, suffix) in [(PieceType::KNIGHT, 'n',), (PieceType::BISHOP, 'b',), (PieceType::ROOK, 'r',), (PieceType::QUEEN, 'q',),] {
            assert_eq!(
                Move::create_move(
                    48,
                    56,
                    promotion,
                    MoveFlag::QUIET,
                ).get_uci(),
                format!("a7a8{suffix}"),
            );
        }

        assert_eq!(
            Move::default().get_uci(),
            "0000",
        );
    }

    #[test]
    fn test_piece_and_color_semantics_are_consistent() {
        for piece in 1..=6 {
            assert_eq!(
                get_piece_type(piece),
                get_piece_type(-piece),
            );
            assert_eq!(
                get_piece_index(-piece),
                get_piece_index(piece) + 6,
            );
            assert_eq!(
                get_piece_score(piece),
                PIECE_SCORES[piece as usize],
            );
            assert_eq!(
                get_piece_color(piece),
                Color::WHITE,
            );
            assert_eq!(
                get_piece_color(-piece),
                Color::BLACK,
            );
        }

        for color in [Color::WHITE, Color::BLACK,] {
            assert_eq!(
                get_opponent(get_opponent(color)),
                color,
            );
            assert_ne!(
                get_color_index(color),
                get_color_index(get_opponent(color)),
            );
        }
    }

    #[test]
    fn test_move_capacity_rejection_preserves_existing_moves() {
        let mut moves = MoveList::default();
        let chess_move = Move::create_move(
            12,
            28,
            PieceType::NONE,
            MoveFlag::DOUBLE_PUSH,
        );

        for _ in 0..MAXIMUM_MOVE_COUNT {
            moves.add_move(chess_move).unwrap();
        }

        assert!(moves.add_move(Move::default()).is_err());
        assert_eq!(
            moves.count,
            MAXIMUM_MOVE_COUNT,
        );
        assert!(moves.moves.iter().all(|&stored_move| stored_move == chess_move));
    }

    #[test]
    fn test_engine_errors_preserve_io_causes() {
        let cause = io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Denied test file access.",
        );
        let engine_error = EngineError::from(cause);
        assert!(engine_error.source().is_some());
        assert!(engine_error.to_string().contains("Denied test file access."));
        assert!(EngineError::create_invalid_input("Rejected test input.").source().is_none());
    }
}
