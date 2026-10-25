use std::time::Instant;

use alekhine::chess::position::Position;
use alekhine::chess::types::{
    EngineResult,
    START_FEN,
};
use alekhine::engine::search::{
    Search,
    SearchControl,
    SearchLimits,
    SearchOptions,
};
use alekhine::engine::transposition_table::TranspositionTable;
use alekhine::evaluation::network::Network;

const BENCHMARK_POSITIONS: [&str; 4] = [
    START_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1",
];
const CHECKSUM_MULTIPLIER: u64 = 1099511628211;


pub fn run_benchmark(depth_plies: i32) -> EngineResult<()> {
    let network = Network::create_network();
    let mut table = TranspositionTable::create_table(16)?;
    let control = SearchControl::default();
    let limits = SearchLimits {
        depth_plies,
        ..SearchLimits::default()
    };
    let mut total_node_count = 0;
    let mut checksum = 0u64;
    let started_at = Instant::now();

    for fen in BENCHMARK_POSITIONS {
        table.clear_table();
        control.start_search(&limits);
        let mut search = Search::create_search(
            &network,
            &mut table,
            &control,
            SearchOptions::default(),
        )?;
        let search_result = search.run_search(
            Position::parse_fen(fen)?,
            &limits,
            &mut |_| (),
        )?;
        total_node_count += search_result.node_count;
        checksum = checksum.wrapping_mul(CHECKSUM_MULTIPLIER) ^ search_result.node_count ^ search_result.best_move.encoding as u64 ^ search_result.score as u64;
    }

    let elapsed_milliseconds = started_at.elapsed().as_millis().max(1);
    println!("Completed benchmark at depth {depth_plies}.");
    println!("Nodes: {total_node_count}.");
    println!("Time: {elapsed_milliseconds} milliseconds.");
    println!(
        "Nodes per second: {}.",
        total_node_count as u128 * 1000 / elapsed_milliseconds,
    );
    println!("Checksum: {checksum}.");
    println!(
        "Network fingerprint: {}.",
        network.get_fingerprint(),
    );

    return Ok(());
}
