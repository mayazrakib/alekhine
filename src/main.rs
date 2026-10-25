use std::env;
use std::io;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use alekhine::chess::position::{
    count_perft,
    Position,
};
use alekhine::chess::types::{
    EngineError,
    EngineResult,
    MAXIMUM_PLY_COUNT,
    START_FEN,
};
use alekhine::evaluation::network::Network;
use alekhine::evaluation::stockfish_network::{
    STOCKFISH_BUCKET_COUNT,
    StockfishNetwork,
};
use alekhine::protocol::uci::UciSession;
use alekhine::training::native::{
    train_network,
    TrainingOptions,
};
use alekhine::training::stockfish::fine_tune_stockfish;

#[path = "../tests/perf/engine_benchmark.rs"]
mod engine_benchmark;

use engine_benchmark::run_benchmark;

fn parse_depth_plies(text: &str) -> EngineResult<i32> {
    let depth_plies = text.parse::<i32>().map_err(|_| EngineError::create_invalid_input("Failed to parse the search depth."))?;

    if text.starts_with('+') || (depth_plies < 0) || (depth_plies >= (MAXIMUM_PLY_COUNT as i32 - 1)) {
        return Err(EngineError::create_invalid_input("Failed to parse the search depth."));
    }

    return Ok(depth_plies);
}

fn run_engine() -> EngineResult<ExitCode> {
    let arguments: Vec<String> = env::args().skip(1).collect();

    if arguments.is_empty() {
        let mut session = UciSession::create_session(
            io::stdin().lock(),
            io::stdout(),
        )?;
        session.run_session()?;

        return Ok(ExitCode::SUCCESS);
    }

    let command = arguments[0].as_str();

    if (command == "version") && (arguments.len() == 1) {
        let network = Network::create_network();
        println!(
            "alekhine {}",
            env!("CARGO_PKG_VERSION"),
        );
        println!("A research-grade chess engine.");
        println!("Author: Mayaz Rakib.");
        println!("Copyright (C) 2026 Mayaz Rakib.");
        println!("License: AGPL-3.0-or-later.");
        println!(
            "Revision: {}.",
            env!("ALEKHINE_REVISION"),
        );
        println!(
            "Compiler: {}.",
            env!("ALEKHINE_COMPILER"),
        );
        println!("Rust edition: 2024.");
        println!(
            "Network fingerprint: {}.",
            network.get_fingerprint(),
        );
        println!("The bundled bootstrap weights are untrained.");
    } else if (command == "bench") && (arguments.len() <= 2) {
        run_benchmark(if arguments.len() == 2 {
            parse_depth_plies(&arguments[1])?
        } else {
            4
        })?;
    } else if ((command == "perft") || (command == "divide")) && ((arguments.len() == 2) || (arguments.len() == 3)) {
        let depth_plies = parse_depth_plies(&arguments[1])?;

        let fen = if arguments.len() == 3 {
            &arguments[2]
        } else {
            START_FEN
        };

        let mut position = Position::parse_fen(fen)?;
        let started_at = Instant::now();
        let mut node_count = 0;

        if (command == "divide") && (depth_plies > 0) {
            let moves = position.generate_legal_moves(false)?;

            for &chess_move in &moves.moves[..moves.count] {
                let undo = position.make_move(chess_move)?;
                let subtree = count_perft(
                    &mut position,
                    depth_plies - 1,
                );
                position.unmake_move(&undo);
                let subtree_count = subtree?;
                node_count += subtree_count;
                println!(
                    "{} {subtree_count}",
                    chess_move.get_uci(),
                );
            }
        } else {
            node_count = count_perft(
                &mut position,
                depth_plies,
            )?;
        }

        let elapsed_milliseconds = started_at.elapsed().as_millis();
        println!("Counted {node_count} leaf nodes in {elapsed_milliseconds} milliseconds.");
    } else if (command == "inspect_stockfish_network") && ((arguments.len() == 2) || (arguments.len() == 3)) {
        let network = StockfishNetwork::load_network(Path::new(&arguments[1]))?;
        let position = Position::parse_fen(if arguments.len() == 3 { &arguments[2] } else { START_FEN })?;
        let board = std::array::from_fn(|square| position.get_piece(square as i32));
        let features = network.build_features(
            &board,
            position.get_side_to_move(),
        );
        println!(
            "{{\"hidden_width\":{},\"network_fingerprint\":{},\"bucket_index\":{},\"centipawn_scale\":{},\"buckets\":[",
            network.get_hidden_width(),
            network.get_fingerprint(),
            features.bucket_index,
            features.centipawn_scale,
        );

        for bucket_index in 0..STOCKFISH_BUCKET_COUNT {
            let evaluation = network.evaluate_features(
                &features,
                bucket_index,
            );
            let separator = if bucket_index + 1 == STOCKFISH_BUCKET_COUNT { "" } else { "," };
            println!(
                "{{\"bucket_index\":{bucket_index},\"psqt_score\":{},\"positional_score\":{},\"centipawn_score\":{}}}{separator}",
                evaluation.psqt_score,
                evaluation.positional_score,
                evaluation.centipawn_score,
            );
        }

        println!("]}}");
    } else if ((command == "train_network") && (arguments.len() >= 3))
        || ((command == "fine_tune_stockfish") && (arguments.len() >= 4)) {
        let is_stockfish_training = command == "fine_tune_stockfish";
        let mut options = TrainingOptions::default();
        let mut index = if is_stockfish_training { 4 } else { 3 };

        if is_stockfish_training {
            options.learning_rate = 0.0001;
        }

        if (index < arguments.len()) && !arguments[index].starts_with("--") {
            options.epoch_count = arguments[index].parse::<usize>()
                .map_err(|_| EngineError::create_invalid_input("Failed to parse the training epoch count."))?;
            index += 1;
        }

        let mut configured_options = Vec::new();

        while index < arguments.len() {
            let name = &arguments[index];
            let option_value = arguments.get(index + 1)
                .ok_or_else(|| EngineError::create_invalid_input("Failed to parse a missing training option value."))?;

            if configured_options.contains(name) {
                return Err(EngineError::create_invalid_input("Failed to parse a duplicate training option."));
            }

            configured_options.push(name.clone());

            match name.as_str() {
                "--batch_size" => options.batch_size = option_value.parse()
                    .map_err(|_| EngineError::create_invalid_input("Failed to parse the training batch size."))?,
                "--learning_rate" => options.learning_rate = option_value.parse()
                    .map_err(|_| EngineError::create_invalid_input("Failed to parse the training learning rate."))?,
                "--seed" => options.seed = option_value.parse()
                    .map_err(|_| EngineError::create_invalid_input("Failed to parse the training seed."))?,
                "--maximum_record_count" => options.maximum_record_count = option_value.parse()
                    .map_err(|_| EngineError::create_invalid_input("Failed to parse the training record budget."))?,
                _ => return Err(EngineError::create_invalid_input("Failed to parse an unsupported training option.")),
            }

            index += 2;
        }

        if is_stockfish_training {
            let report = fine_tune_stockfish(
                Path::new(&arguments[1]),
                Path::new(&arguments[2]),
                Path::new(&arguments[3]),
                &options,
                &mut |message| println!("{message}"),
            )?;
            println!(
                "Saved Stockfish network fingerprint {} from epoch {} with {} changed parameters; validation loss changed from {:.6} to {:.6}.",
                report.network_fingerprint,
                report.best_epoch,
                report.changed_parameter_count,
                report.initial_validation_loss,
                report.best_validation_loss,
            );
        } else {
            let report = train_network(
                Path::new(&arguments[1]),
                Path::new(&arguments[2]),
                &options,
                &mut |message| println!("{message}"),
            )?;
            println!(
                "Saved network fingerprint {} with best validation loss {:.6}; initial loss was {:.6}.",
                report.network_fingerprint,
                report.best_validation_loss,
                report.initial_validation_loss,
            );
            println!(
                "Maximum sampled quantization error: {:.2} centipawns.",
                report.maximum_quantization_error_centipawns,
            );
        }
    } else if (command == "export_network") && (arguments.len() == 2) {
        Network::create_network().save_network(Path::new(&arguments[1]))?;
        println!("Exported the untrained bootstrap network.");
    } else {
        eprintln!("Use alekhine with no arguments for UCI, or use version, bench [depth], perft depth [FEN], divide depth [FEN], export_network path, train_network dataset output [epochs], inspect_stockfish_network model [FEN], or fine_tune_stockfish model dataset output [epochs].");

        return Ok(ExitCode::from(2));
    }

    return Ok(ExitCode::SUCCESS);
}

fn main() -> ExitCode {
    match run_engine() {
        Ok(exit_code) => return exit_code,
        Err(error) => {
            eprintln!("{error}");

            return ExitCode::FAILURE;
        }
    }
}
