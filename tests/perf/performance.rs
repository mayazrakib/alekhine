use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::{
    BufWriter,
    Write,
};
use std::path::Path;
use std::time::Instant;

use alekhine::chess::position::{
    count_perft,
    Position,
};
use alekhine::chess::tactical::calculate_exchange_score;
use alekhine::chess::types::START_FEN;
use alekhine::engine::search::{
    Search,
    SearchControl,
    SearchLimits,
    SearchOptions,
    SearchResult,
};
use alekhine::engine::transposition_table::{
    BoundType,
    TranspositionTable,
};
use alekhine::evaluation::network::Network;
use alekhine::evaluation::neural::encode_features;

type BenchmarkResult<T> = Result<
    T,
    Box<dyn Error>,
>;

const MAXIMUM_ITERATION_COUNT: u64 = 1_048_576;
const CHECKSUM_MULTIPLIER: u64 = 1_099_511_628_211;
const SPECIAL_POSITIONS: [&str; 5] = [
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
];

struct BenchmarkOptions {
    sample_count: usize,
    minimum_milliseconds: u128,
    node_count: u64,
    depth_plies: i32,
    network_path: Option<String>,
    suite: String,
    dataset_path: Option<String>,
    record_count: usize,
}

fn parse_options() -> BenchmarkResult<BenchmarkOptions> {
    let mut options = BenchmarkOptions {
        sample_count: 5,
        minimum_milliseconds: 15,
        node_count: 2000,
        depth_plies: 3,
        network_path: None,
        suite: String::from("all"),
        dataset_path: None,
        record_count: 256,
    };
    let mut arguments = env::args().skip(1);

    while let Some(argument) = arguments.next() {
        if argument == "--bench" {
            continue;
        }

        let parameter = arguments.next().ok_or("Failed to parse a missing benchmark argument.")?;

        match argument.as_str() {
            "--samples" => options.sample_count = parameter.parse()?,
            "--minimum_ms" => options.minimum_milliseconds = parameter.parse()?,
            "--nodes" => options.node_count = parameter.parse()?,
            "--depth" => options.depth_plies = parameter.parse()?,
            "--network" => options.network_path = Some(parameter),
            "--suite" => options.suite = parameter,
            "--dataset" => options.dataset_path = Some(parameter),
            "--records" => options.record_count = parameter.parse()?,
            _ => return Err("Failed to parse an unknown benchmark argument.".into()),
        }
    }

    if !(1..=100).contains(&options.sample_count)
        || !(1..=1000).contains(&options.minimum_milliseconds)
        || !(32..=1_000_000).contains(&options.node_count)
        || !(1..=6).contains(&options.depth_plies)
        || !(32..=100_000).contains(&options.record_count)
        || !["all", "micro", "search",].contains(&options.suite.as_str())
    {
        return Err("Failed to validate bounded benchmark options.".into());
    }

    return Ok(options);
}

fn load_positions() -> BenchmarkResult<Vec<Position>> {
    let mut positions = Vec::new();

    for line in include_str!("../../test/openings.epd").lines() {
        let fields: Vec<&str> = line.split_whitespace().take(4).collect();
        positions.push(Position::parse_fen(&format!(
            "{} 0 1",
            fields.join(" "),
        ))?);
    }

    for fen in SPECIAL_POSITIONS {
        positions.push(Position::parse_fen(fen)?);
    }

    return Ok(positions);
}

fn measure_operation<F>(
    operation: &mut F,
    iteration_count: u64,
) -> BenchmarkResult<(
    u128,
    u64,
)>
where
    F: FnMut(u64) -> BenchmarkResult<u64>,
{
    let started_at = Instant::now();
    let mut checksum = 0_u64;

    for index in 0..iteration_count {
        checksum = checksum.wrapping_mul(CHECKSUM_MULTIPLIER) ^ black_box(operation(index)?);
    }

    return Ok((started_at.elapsed().as_nanos(), checksum,));
}

fn write_sample(
    name: &str,
    model: &str,
    phase: &str,
    sample_index: usize,
    operation_count: u64,
    elapsed_nanoseconds: u128,
    checksum: u64,
) {
    println!(
        "{{\"suite\":\"micro\",\"name\":\"{name}\",\"model\":\"{model}\",\"phase\":\"{phase}\",\"sample_index\":{sample_index},\"operation_count\":{operation_count},\"elapsed_nanoseconds\":{elapsed_nanoseconds},\"checksum\":{checksum}}}",
    );
}

fn run_case<F>(
    name: &str,
    model: &str,
    options: &BenchmarkOptions,
    mut operation: F,
) -> BenchmarkResult<()>
where
    F: FnMut(u64) -> BenchmarkResult<u64>,
{
    let (elapsed_nanoseconds, checksum,) = measure_operation(
        &mut operation,
        1,
    )?;
    write_sample(
        name,
        model,
        "first pass",
        0,
        1,
        elapsed_nanoseconds,
        checksum,
    );
    let mut iteration_count = 1;

    while iteration_count < MAXIMUM_ITERATION_COUNT {
        let (elapsed_nanoseconds, _,) = measure_operation(
            &mut operation,
            iteration_count,
        )?;

        if elapsed_nanoseconds >= options.minimum_milliseconds * 1_000_000 {
            break;
        }

        iteration_count *= 2;
    }

    for _ in 0..2 {
        black_box(measure_operation(
            &mut operation,
            iteration_count,
        )?);
    }

    for sample_index in 0..options.sample_count {
        let (elapsed_nanoseconds, checksum,) = measure_operation(
            &mut operation,
            iteration_count,
        )?;
        write_sample(
            name,
            model,
            "measured",
            sample_index,
            iteration_count,
            elapsed_nanoseconds,
            checksum,
        );
    }

    return Ok(());
}

fn run_microbenchmarks(
    network: &Network,
    model: &str,
    options: &BenchmarkOptions,
) -> BenchmarkResult<()> {
    let mut positions = load_positions()?;
    let position_count = positions.len();
    let fens: Vec<String> = positions.iter().map(Position::get_fen).collect();
    let accumulators: Vec<_> = positions.iter().map(|position| network.build_accumulator(position)).collect();
    let mut moves = Vec::new();
    let mut changes = Vec::new();
    let mut boards = Vec::new();

    for (index, position,) in positions.iter_mut().enumerate() {
        let legal_moves = position.generate_legal_moves(false)?;
        let undo = position.make_move(legal_moves.moves[0])?;
        assert_eq!(
            network.update_accumulator(
                &accumulators[index],
                &undo,
            ),
            network.build_accumulator(position),
        );
        position.unmake_move(&undo);
        changes.push(undo);
        moves.push(legal_moves);
        boards.push(std::array::from_fn(|square| position.get_piece(square as i32)));
        assert_eq!(
            position.get_fen(),
            fens[index],
        );
        assert!(position.has_valid_state());
    }

    run_case(
        "parse_fen",
        model,
        options,
        |index| Ok(black_box(Position::parse_fen(black_box(&fens[index as usize % position_count]))?).get_key()),
    )?;
    run_case(
        "serialize_fen",
        model,
        options,
        |index| Ok(black_box(positions[index as usize % position_count].get_fen()).len() as u64),
    )?;

    for (name, is_tactical,) in [("legal_moves", false,), ("tactical_moves", true,),] {
        run_case(
            name,
            model,
            options,
            |index| Ok(black_box(positions[index as usize % position_count].generate_legal_moves(is_tactical)?).count as u64),
        )?;
    }

    run_case(
        "has_legal_move",
        model,
        options,
        |index| Ok(black_box(positions[index as usize % position_count].has_legal_move()?) as u64),
    )?;
    run_case(
        "make_unmake",
        model,
        options,
        |index| {
            let slot = index as usize % position_count;
            let undo = positions[slot].make_move(black_box(moves[slot].moves[0]))?;
            let key = black_box(positions[slot].get_key());
            positions[slot].unmake_move(&undo);

            return Ok(key);
        },
    )?;
    run_case(
        "accumulator_full",
        model,
        options,
        |index| Ok(black_box(network.build_accumulator(black_box(&positions[index as usize % position_count]))).bootstrap[0] as u64),
    )?;
    run_case(
        "accumulator_incremental",
        model,
        options,
        |index| {
            let slot = index as usize % position_count;
            let accumulator = network.update_accumulator(
                black_box(&accumulators[slot]),
                black_box(&changes[slot]),
            );

            return Ok(black_box(accumulator).bootstrap[0] as u64);
        },
    )?;
    run_case(
        "neural_features",
        model,
        options,
        |index| Ok(black_box(encode_features(black_box(&boards[index as usize % position_count]))).counts[0] as u64),
    )?;
    run_case(
        "evaluate_value",
        model,
        options,
        |index| {
            let slot = index as usize % position_count;

            return Ok(black_box(network.evaluate_position(
                black_box(&positions[slot]),
                black_box(&accumulators[slot]),
            )) as u64);
        },
    )?;
    run_case(
        "evaluate_policy_batch",
        model,
        options,
        |index| {
            let slot = index as usize % position_count;
            let policies = network.evaluate_policies(
                black_box(&positions[slot]),
                black_box(&accumulators[slot]),
                black_box(&moves[slot]),
            );

            return Ok(black_box(policies)[0].to_bits());
        },
    )?;
    run_case(
        "static_exchange",
        model,
        options,
        |index| {
            let slot = index as usize % position_count;
            let move_index = index as usize % moves[slot].count;

            return Ok(black_box(calculate_exchange_score(
                black_box(&positions[slot]),
                black_box(moves[slot].moves[move_index]),
            )) as u64);
        },
    )?;
    let mut table = TranspositionTable::create_table(16)?;

    run_case(
        "transposition_store_probe",
        model,
        options,
        |index| {
            let key = index.wrapping_mul(CHECKSUM_MULTIPLIER);
            table.store_position(
                black_box(key),
                moves[0].moves[0],
                50,
                4,
                BoundType::EXACT,
                0,
            );
            let entry = black_box(table.probe_position(black_box(key))).ok_or("Failed to recover a benchmark transposition.")?;

            return Ok(entry.key);
        },
    )?;

    for (slot, expected_count,) in [(0, 8902,), (4, 97862,), (5, 2812,),] {
        let actual_count = count_perft(
            &mut positions[slot],
            3,
        )?;
        assert_eq!(
            actual_count,
            expected_count,
        );
        run_case(
            &format!(
                "perft_3_position_{slot}",
            ),
            model,
            options,
            |_| Ok(black_box(count_perft(
                &mut positions[slot],
                3,
            )?)),
        )?;
    }

    for (index, position,) in positions.iter().enumerate() {
        assert_eq!(
            position.get_fen(),
            fens[index],
        );
        assert!(position.has_valid_state());
    }

    return Ok(());
}

fn execute_search(
    position: &Position,
    network: &Network,
    table: &mut TranspositionTable,
    options: SearchOptions,
    limits: &SearchLimits,
) -> BenchmarkResult<SearchResult> {
    let control = SearchControl::default();
    control.start_search(limits);
    let mut search = Search::create_search(
        network,
        table,
        &control,
        options,
    )?;

    return Ok(search.run_search(
        position.clone(),
        limits,
        &mut |_| {},
    )?);
}

fn run_search_benchmarks(
    network: &Network,
    model: &str,
    options: &BenchmarkOptions,
) -> BenchmarkResult<()> {
    let mut positions = load_positions()?;
    let profiles = ["hybrid", "policy_off", "full_search", "multipv_3", "hash_1", "hash_64",];

    for profile in profiles {
        let mut search_options = SearchOptions::default();
        let hash_mib = match profile {
            "hash_1" => 1,
            "hash_64" => 64,
            _ => 16,
        };

        match profile {
            "policy_off" => search_options.puct_simulation_count = 0,
            "full_search" => {
                search_options.puct_simulation_count = 0;
                search_options.is_selective_search_enabled = false;
            },
            "multipv_3" => search_options.variation_count = 3,
            _ => {},
        }

        let mut table = TranspositionTable::create_table(hash_mib)?;

        for (position_index, position,) in positions.iter_mut().enumerate() {
            let legal_moves = position.generate_legal_moves(false)?;

            for budget in ["nodes", "depth",] {
                let limits = if budget == "nodes" {
                    SearchLimits {
                        node_count: options.node_count,
                        ..SearchLimits::default()
                    }
                } else {
                    SearchLimits {
                        depth_plies: options.depth_plies,
                        ..SearchLimits::default()
                    }
                };

                for hash_state in ["cleared", "warm",] {
                    for sample_index in 0..(options.sample_count + 1) {
                        table.clear_table();

                        if hash_state == "warm" {
                            black_box(execute_search(
                                position,
                                network,
                                &mut table,
                                search_options,
                                &limits,
                            )?);
                        }

                        let started_at = Instant::now();
                        let search_result = execute_search(
                            position,
                            network,
                            &mut table,
                            search_options,
                            &limits,
                        )?;
                        let elapsed_nanoseconds = started_at.elapsed().as_nanos();
                        assert!(legal_moves.moves[..legal_moves.count].contains(&search_result.best_move));
                        assert!((limits.node_count == 0) || (search_result.node_count <= limits.node_count));
                        let phase = if sample_index == 0 { "first pass" } else { "measured" };
                        let best_move = search_result.best_move.get_uci();
                        let node_count = search_result.node_count;
                        let depth_plies = search_result.completed_depth_plies;
                        let score = search_result.score;
                        println!(
                            "{{\"suite\":\"search\",\"name\":\"{profile}_position_{position_index}_{budget}_{hash_state}\",\"model\":\"{model}\",\"phase\":\"{phase}\",\"sample_index\":{sample_index},\"operation_count\":1,\"elapsed_nanoseconds\":{elapsed_nanoseconds},\"node_count\":{node_count},\"completed_depth_plies\":{depth_plies},\"best_move\":\"{best_move}\",\"score_centipawns\":{score},\"hash_mib\":{hash_mib}}}",
                        );
                    }
                }
            }
        }
    }

    return Ok(());
}

fn generate_dataset(
    path: &Path,
    record_count: usize,
) -> BenchmarkResult<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    let mut position = Position::parse_fen(START_FEN)?;
    let mut position_keys = HashSet::new();
    let mut seed = 20261004_u64;
    let mut game_index = 0;
    let mut game_ply_count = 0;
    let mut written_count = 0;

    for _ in 0..(record_count * 100) {
        let legal_moves = position.generate_legal_moves(false)?;

        if (legal_moves.count == 0) || position.is_rule_draw() || (game_ply_count >= 16) {
            position = Position::parse_fen(START_FEN)?;
            game_index += 1;
            game_ply_count = 0;

            continue;
        }

        let fen = position.get_fen();
        let identity = fen.split_whitespace().take(4).collect::<Vec<_>>().join(" ");

        if position_keys.insert(identity) {
            let game_id = match game_index {
                0 => String::from("game_0"),
                1 => String::from("game_3"),
                _ => format!(
                    "performance_game_{game_index}",
                ),
            };
            writeln!(
                writer,
                "{fen} | {} | 0.5 | {}:1 | {game_id}",
                position.get_material_score(),
                legal_moves.moves[0].get_uci(),
            )?;
            written_count += 1;

            if written_count == record_count {
                writer.flush()?;

                return Ok(());
            }
        }

        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        position.make_move(legal_moves.moves[seed as usize % legal_moves.count])?;
        game_ply_count += 1;
    }

    return Err("Failed to generate the bounded synthetic benchmark dataset.".into());
}

fn run_benchmarks() -> BenchmarkResult<()> {
    let options = parse_options()?;

    if let Some(path) = &options.dataset_path {
        return generate_dataset(
            Path::new(path),
            options.record_count,
        );
    }

    for (position_index, position,) in load_positions()?.iter().enumerate() {
        println!(
            "{{\"suite\":\"metadata\",\"position_index\":{position_index},\"fen\":\"{}\"}}",
            position.get_fen(),
        );
    }

    let mut networks = vec![("bootstrap", Network::create_network(),),];

    if let Some(path) = &options.network_path {
        networks.push(("nnue", Network::load_network(Path::new(path))?,));
    }

    for (model, network,) in networks {
        println!(
            "{{\"suite\":\"metadata\",\"model\":\"{model}\",\"network_fingerprint\":{}}}",
            network.get_fingerprint(),
        );

        if options.suite != "search" {
            run_microbenchmarks(
                &network,
                model,
                &options,
            )?;
        }

        if options.suite != "micro" {
            run_search_benchmarks(
                &network,
                model,
                &options,
            )?;
        }
    }

    return Ok(());
}

fn main() {
    if let Err(error) = run_benchmarks() {
        eprintln!(
            "Failed to complete performance testing: {error}",
        );
        std::process::exit(1);
    }
}
