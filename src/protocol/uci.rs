use std::io::{
    BufRead,
    Write,
};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{
    Arc,
    Condvar,
    Mutex,
};
use std::thread::{
    Builder,
    JoinHandle,
};

use crate::chess::position::Position;
use crate::chess::types::{
    EngineError,
    EngineResult,
    MATE_SCORE,
    MATE_THRESHOLD,
    MAXIMUM_HISTORY_COUNT,
    MAXIMUM_PLY_COUNT,
    Move,
    START_FEN,
};
use crate::engine::search::{
    Search,
    SearchControl,
    SearchOptions,
    SearchReport,
    TimeBudget,
};
use crate::engine::transposition_table::TranspositionTable;
use crate::evaluation::network::Network;
use crate::protocol::commands::{
    parse_integer,
    parse_search_limits,
};

const MAXIMUM_COMMAND_BYTES: usize = 65536;
const SEARCH_STACK_BYTES: usize = 4 * 1024 * 1024;
type CompletionSignal = (Mutex<()>, Condvar,);

fn write_line<O: Write>(
    output: &Mutex<O>,
    line: &str,
) -> EngineResult<()> {
    let mut writer = output.lock().map_err(|_| EngineError::create_invalid_input("Failed to lock protocol output."))?;
    writeln!(
        writer,
        "{line}",
    )?;
    writer.flush()?;

    return Ok(());
}

fn report_search<O: Write>(
    output: &Mutex<O>,
    report: &SearchReport,
) -> EngineResult<()> {
    let mut line = format!(
        "info depth {} seldepth {} multipv {}",
        report.depth_plies,
        report.selective_depth_plies,
        report.variation_index,
    );

    if report.score.abs() >= MATE_THRESHOLD {
        let mate_ply_count = MATE_SCORE - report.score.abs();
        let mate_move_count = (mate_ply_count + 1) / 2;

        let signed_mate_count = if report.score >= 0 {
            mate_move_count
        } else {
            -mate_move_count
        };

        line.push_str(&format!(" score mate {signed_mate_count}"));
    } else {
        line.push_str(&format!(
            " score cp {}",
            report.score,
        ));
    }

    let elapsed_milliseconds = report.elapsed_milliseconds.max(1);
    line.push_str(&format!(
        " nodes {} nps {} time {} hashfull {} tbhits 0 pv",
        report.node_count,
        (report.node_count * 1000) / elapsed_milliseconds as u64,
        report.elapsed_milliseconds,
        report.hashfull_per_mille,
    ));

    for chess_move in &report.principal_variation {
        line.push(' ');
        line.push_str(&chess_move.get_uci());
    }

    return write_line(
        output,
        &line,
    );
}

pub struct UciSession<
    I: BufRead,
    O: Write + Send + 'static,
> {
    input: I,
    output: Arc<Mutex<O>>,
    position: Position,
    network: Arc<Network>,
    transposition_table: Arc<Mutex<TranspositionTable>>,
    control: Arc<SearchControl>,
    options: SearchOptions,
    active_time_budget: TimeBudget,
    move_overhead_milliseconds: i64,
    completion: Arc<CompletionSignal>,
    worker: Option<JoinHandle<EngineResult<()>>>,
}

impl<
    I: BufRead,
    O: Write + Send + 'static,
> UciSession<
    I,
    O,
> {
    pub fn create_session(
        command_input: I,
        protocol_output: O,
    ) -> EngineResult<Self> {
        return Ok(Self {
            input: command_input,
            output: Arc::new(Mutex::new(protocol_output)),
            position: Position::parse_fen(START_FEN)?,
            network: Arc::new(Network::create_network()),
            transposition_table: Arc::new(Mutex::new(TranspositionTable::create_table(16)?)),
            control: Arc::new(SearchControl::default()),
            options: SearchOptions::default(),
            active_time_budget: TimeBudget::default(),
            move_overhead_milliseconds: 20,
            completion: Arc::new((Mutex::new(()), Condvar::new(),)),
            worker: None,
        });
    }

    fn stop_search(&mut self) -> EngineResult<()> {
        {
            let completion_guard = self.completion.0.lock().map_err(|_| EngineError::create_invalid_input("Failed to lock search completion."))?;
            self.control.should_stop.store(
                true,
                Ordering::SeqCst,
            );
            drop(completion_guard);
        }

        self.completion.1.notify_all();

        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| EngineError::create_invalid_input("Failed to join the search worker because it panicked."))??;
        }

        return Ok(());
    }

    fn read_command(&mut self) -> EngineResult<Option<String>> {
        let mut command = Vec::with_capacity(256);
        let mut is_oversized = false;
        let mut has_read_bytes = false;

        loop {
            let buffer = self.input.fill_buf()?;

            if buffer.is_empty() {
                break;
            }

            has_read_bytes = true;
            let newline_index = buffer.iter().position(|&symbol| symbol == b'\n');
            let content_count = newline_index.unwrap_or(buffer.len());
            let available_count = MAXIMUM_COMMAND_BYTES - command.len();
            let retained_count = content_count.min(available_count);
            command.extend_from_slice(&buffer[..retained_count]);
            is_oversized = is_oversized || (content_count > available_count);
            let consumed_count = content_count + usize::from(newline_index.is_some());
            self.input.consume(consumed_count);

            if newline_index.is_some() {
                break;
            }
        }

        if !has_read_bytes {
            return Ok(None);
        }

        if is_oversized {
            write_line(
                &self.output,
                "info string Rejected a command exceeding 65,536 bytes.",
            )?;

            return Ok(Some(String::new()));
        }

        let command = String::from_utf8(command).map_err(|_| EngineError::create_invalid_input("Failed to decode a UCI command as UTF-8."))?;

        return Ok(Some(command));
    }

    fn set_position(
        &mut self,
        tokens: &[&str],
    ) -> EngineResult<()> {
        self.stop_search()?;

        if tokens.len() < 2 {
            return Err(EngineError::create_invalid_input("Failed to set the position because its arguments are missing."));
        }

        let mut candidate;
        let mut index;

        if tokens[1] == "startpos" {
            candidate = Position::parse_fen(START_FEN)?;
            index = 2;
        } else if (tokens[1] == "fen") && (tokens.len() >= 8) {
            candidate = Position::parse_fen(&tokens[2..8].join(" "))?;
            index = 8;
        } else {
            return Err(EngineError::create_invalid_input("Failed to set the position because its FEN arguments are invalid."));
        }

        if index < tokens.len() {
            if tokens[index] != "moves" {
                return Err(EngineError::create_invalid_input("Failed to set the position because a moves marker was expected."));
            }

            index += 1;

            if (tokens.len() - index) >= (MAXIMUM_HISTORY_COUNT - MAXIMUM_PLY_COUNT - 1) {
                return Err(EngineError::create_invalid_input("Failed to set the position because its move history exceeds the supported capacity."));
            }

            for move_name in &tokens[index..] {
                let chess_move = candidate.find_legal_move(move_name)?;
                candidate.make_move(chess_move)?;
            }
        }

        self.position = candidate;

        return Ok(());
    }

    fn set_option(
        &mut self,
        tokens: &[&str],
    ) -> EngineResult<()> {
        self.stop_search()?;

        if (tokens.len() < 3) || (tokens[1] != "name") {
            return Err(EngineError::create_invalid_input("Failed to set an option because its name is missing."));
        }

        let name_end = tokens[2..].iter().position(|&token| token == "value").map_or(
            tokens.len(),
            |index| index + 2,
        );
        let name = tokens[2..name_end].join(" ");

        let option_value = if name_end == tokens.len() {
            String::new()
        } else {
            tokens[(name_end + 1)..].join(" ")
        };

        match name.as_str() {
            "Hash" => {
                let size_mib = parse_integer(
                    &option_value,
                    1,
                    512,
                )? as usize;
                self.transposition_table.lock().map_err(|_| EngineError::create_invalid_input("Failed to lock the hash table."))?.resize_table(size_mib)?;
            }
            "Clear Hash" => self.transposition_table.lock().map_err(|_| EngineError::create_invalid_input("Failed to lock the hash table."))?.clear_table(),
            "Threads" => {
                parse_integer(
                    &option_value,
                    1,
                    1,
                )?;
            }
            "MultiPV" => self.options.variation_count = parse_integer(
                &option_value,
                1,
                16,
            )? as usize,
            "Policy Simulations" => self.options.puct_simulation_count = parse_integer(
                &option_value,
                0,
                4096,
            )? as usize,
            "policy_verification_depth_plies" => self.options.policy_verification_depth_plies = parse_integer(
                &option_value,
                0,
                3,
            )? as i32,
            "policy_uncertainty_percent" => self.options.policy_uncertainty_percent = parse_integer(
                &option_value,
                0,
                100,
            )? as i32,
            "aggression_margin_centipawns" => self.options.aggression_margin_centipawns = parse_integer(
                &option_value,
                0,
                100,
            )? as i32,
            "is_selective_search_enabled" => {
                if (option_value != "true") && (option_value != "false") {
                    return Err(EngineError::create_invalid_input("Failed to set is_selective_search_enabled because its value must be true or false."));
                }

                self.options.is_selective_search_enabled = option_value == "true";
            }
            "Move Overhead" => self.move_overhead_milliseconds = parse_integer(
                &option_value,
                0,
                5000,
            )?,
            "Ponder" => {
                if (option_value != "true") && (option_value != "false") {
                    return Err(EngineError::create_invalid_input("Failed to set Ponder because its value must be true or false."));
                }
            }
            "EvalFile" => {
                let replacement = if option_value == "<bootstrap>" {
                    Network::create_network()
                } else {
                    Network::load_network(Path::new(&option_value))?
                };

                self.network = Arc::new(replacement);
                self.transposition_table.lock().map_err(|_| EngineError::create_invalid_input("Failed to lock the hash table."))?.clear_table();

                let training_status = if self.network.is_trained() {
                    "with weights marked as trained."
                } else {
                    "with untrained bootstrap weights."
                };

                write_line(
                    &self.output,
                    &format!(
                        "info string Loaded network fingerprint {} {training_status}",
                        self.network.get_fingerprint(),
                    ),
                )?;
            }
            _ => return Err(EngineError::create_invalid_input("Failed to set an unsupported engine option.")),
        }

        return Ok(());
    }

    fn start_search(
        &mut self,
        tokens: &[&str],
    ) -> EngineResult<()> {
        self.stop_search()?;
        let limits = parse_search_limits(
            tokens,
            &mut self.position,
            self.move_overhead_milliseconds,
        )?;
        self.active_time_budget = limits.time;
        self.control.start_search(&limits);
        let root = self.position.clone();
        let network = Arc::clone(&self.network);
        let table = Arc::clone(&self.transposition_table);
        let control = Arc::clone(&self.control);
        let completion = Arc::clone(&self.completion);
        let output = Arc::clone(&self.output);
        let options = self.options;
        self.worker = Some(Builder::new().name("alekhine_search".to_owned()).stack_size(SEARCH_STACK_BYTES).spawn(move || {
            let searched_result = (|| {
                let mut table_guard = table.lock().map_err(|_| EngineError::create_invalid_input("Failed to lock the hash table."))?;
                let mut search = Search::create_search(
                    &network,
                    &mut table_guard,
                    &control,
                    options,
                )?;
                let mut output_error = None;
                let search_result = search.run_search(
                    root.clone(),
                    &limits,
                    &mut |report| {
                        if output_error.is_none() {
                            if let Err(error) = report_search(
                                &output,
                                report,
                            ) {
                                output_error = Some(error);
                                control.should_stop.store(
                                    true,
                                    Ordering::SeqCst,
                                );
                            }
                        }
                    },
                )?;

                if let Some(error) = output_error {
                    return Err(error);
                }

                return Ok(search_result);
            })();

            let search_result = match searched_result {
                Ok(completed_result) => completed_result,
                Err(error) => {
                    write_line(
                        &output,
                        &format!("info string Search failed: {error}"),
                    )?;
                    let mut fallback = root;
                    let moves = fallback.generate_legal_moves(false)?;
                    let best_move = moves.moves[..moves.count].iter().copied()
                        .find(|chess_move| limits.root_moves.is_empty() || limits.root_moves.contains(chess_move))
                        .unwrap_or(Move::default());
                    write_line(
                        &output,
                        &format!(
                            "bestmove {}",
                            best_move.get_uci(),
                        ),
                    )?;

                    return Ok(());
                }
            };

            let mut completion_guard = completion.0.lock().map_err(|_| EngineError::create_invalid_input("Failed to lock search completion."))?;

            while !control.should_stop.load(Ordering::SeqCst) && (control.is_pondering.load(Ordering::SeqCst) || limits.is_infinite) {
                completion_guard = completion.1.wait(completion_guard).map_err(|_| EngineError::create_invalid_input("Failed to wait for search completion."))?;
            }

            drop(completion_guard);
            let mut bestmove = format!(
                "bestmove {}",
                search_result.best_move.get_uci(),
            );

            if search_result.principal_variation.len() >= 2 {
                bestmove.push_str(&format!(
                    " ponder {}",
                    search_result.principal_variation[1].get_uci(),
                ));
            }

            return write_line(
                &output,
                &bestmove,
            );
        })?);

        return Ok(());
    }

    fn execute_command(
        &mut self,
        tokens: &[&str],
    ) -> EngineResult<bool> {
        match tokens[0] {
            "uci" => {
                let identity = format!(
                    "id name alekhine {}",
                    env!("CARGO_PKG_VERSION"),
                );

                for line in [
                    identity.as_str(),
                    "id author Mayaz Rakib",
                    "option name Hash type spin default 16 min 1 max 512",
                    "option name Clear Hash type button",
                    "option name Threads type spin default 1 min 1 max 1",
                    "option name MultiPV type spin default 1 min 1 max 16",
                    "option name Ponder type check default false",
                    "option name Move Overhead type spin default 20 min 0 max 5000",
                    "option name Policy Simulations type spin default 128 min 0 max 4096",
                    "option name policy_verification_depth_plies type spin default 1 min 0 max 3",
                    "option name policy_uncertainty_percent type spin default 25 min 0 max 100",
                    "option name aggression_margin_centipawns type spin default 15 min 0 max 100",
                    "option name is_selective_search_enabled type check default true",
                    "option name EvalFile type string default <bootstrap>",
                    "uciok",
                ] {
                    write_line(
                        &self.output,
                        line,
                    )?;
                }
            }
            "isready" => write_line(
                &self.output,
                "readyok",
            )?,
            "ucinewgame" => {
                self.stop_search()?;
                self.transposition_table.lock().map_err(|_| EngineError::create_invalid_input("Failed to lock the hash table."))?.clear_table();
                self.position = Position::parse_fen(START_FEN)?;
            }
            "position" => self.set_position(tokens)?,
            "setoption" => self.set_option(tokens)?,
            "go" => self.start_search(tokens)?,
            "stop" => self.stop_search()?,
            "ponderhit" => {
                let completion_guard = self.completion.0.lock().map_err(|_| EngineError::create_invalid_input("Failed to lock search completion."))?;

                if self.control.is_pondering.load(Ordering::SeqCst) {
                    self.control.activate_clock(self.active_time_budget);
                }

                drop(completion_guard);
                self.completion.1.notify_all();
            }
            "quit" => {
                self.stop_search()?;

                return Ok(false);
            }
            "debug" => (),
            _ => write_line(
                &self.output,
                "info string Ignored an unsupported command.",
            )?,
        }

        return Ok(true);
    }

    pub fn run_session(&mut self) -> EngineResult<()> {
        loop {
            let command = match self.read_command() {
                Ok(Some(command_text)) => command_text,
                Ok(None) => break,
                Err(EngineError::INVALID_INPUT(message)) => {
                    write_line(
                        &self.output,
                        &format!("info string {message}"),
                    )?;

                    continue;
                }
                Err(error) => return Err(error),
            };

            let tokens: Vec<&str> = command.split_whitespace().collect();

            if tokens.is_empty() {
                continue;
            }

            match self.execute_command(&tokens) {
                Ok(true) => (),
                Ok(false) => return Ok(()),
                Err(error) => write_line(
                    &self.output,
                    &format!("info string {error}"),
                )?,
            }
        }

        return self.stop_search();
    }
}

impl<
    I: BufRead,
    O: Write + Send + 'static,
> Drop for UciSession<
    I,
    O,
> {
    fn drop(&mut self) {
        if let Err(error) = self.stop_search() {
            eprintln!("Failed to shut down the UCI session: {error}");
        }
    }
}
