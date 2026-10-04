# Alekhine

`alekhine` is a Rust engine for standard chess that speaks the Universal Chess Interface (UCI) protocol. Alpha-beta search chooses the move, with a bounded policy-guided search to help order candidates. The built-in neural network has untrained weights. Alekhine has not shown a playing-strength advantage over Stockfish.

## Build and run

Alekhine requires Rust 1.85 or newer. The engine uses only Rust's standard library. Build and start it from the project directory:
```sh
cargo build --offline --locked --release
./target/release/alekhine
```

With no arguments, the binary reads UCI commands from standard input. For example:
```text
uci
isready
position startpos moves e2e4 e7e5
go depth 4
```

Wait for `bestmove` before sending `quit`. Send `uci` to see the full option list. `Hash`, `MultiPV`, `Ponder`, `Policy Simulations`, and `EvalFile` are available. `Threads` currently accepts one worker. Setting `Policy Simulations` to zero disables tree exploration.

The binary also provides `version`, `perft`, `divide`, and `bench` commands. For example, `./target/release/alekhine perft 5` counts legal leaf positions from the initial board.

The development build script writes its executable to `build/release/alekhine` and records build details:
```sh
python3 scripts/build.py
```

The crate also exposes a Rust library. After a development build, `python3 scripts/package.py` creates executable archives in `dist/` and refuses to overwrite existing archives.

## Engine and networks

Alekhine uses bitboards, legal move generation, reversible moves, and incremental hashes. It handles castling, en passant, promotion, repetition, the fifty-move rule, stalemate, and basic dead positions.

Search uses iterative deepening, quiescence, and selective pruning. The policy-guided tree receives at most 20 percent of the node or hard-time budget. Both searches share the same board state and draw rules. The engine supports multiple principal variations and clocked UCI searches.

The built-in evaluator is an efficiently updatable neural network (NNUE) with quantized, untrained weights. Alekhine can train a larger network with value, outcome, and move-policy heads. It can also load supported Stockfish 18 large and small networks as a separate evaluation backend. Stockfish network evaluation recomputes its features for each position. No external engine source code or network weights are bundled.

Load a supported network through UCI with `setoption name EvalFile value /absolute/path/to/model.nnue`. The bootstrap model remains available with `setoption name EvalFile value <bootstrap>`.

## Training

The Rust trainer accepts records containing a Forsyth-Edwards notation (FEN) position, a White-relative score, an optional game outcome, optional move probabilities, and a game identifier. It separates complete games between training and validation, deduplicates positions, and writes a network and a `.training.json` manifest. It refuses to overwrite existing outputs.

Train an existing dataset with:
```sh
mkdir -p build/training
./build/release/alekhine train_network \
    build/training/positions.txt \
    build/training/model.nnue \
    10
```

To create teacher labels from Portable Game Notation (PGN) games, use `scripts/generate_training_data.py` with Stockfish and `python-chess`. The `fine_tune_stockfish` command accepts a supported Stockfish 18 `.nnue` file, the same dataset format, an output path, and an epoch count. It trains dense layers while keeping the feature transformer and piece-square weights fixed. Other Stockfish network architectures are rejected. Validation loss alone does not measure playing strength.

## Tests and benchmarks

Run the local Rust and Python test suites with:
```sh
python3 scripts/build.py --test
```

The optional Stockfish integration tests require Stockfish 18 and `python-chess`:
```sh
python3 scripts/build.py --test --stockfish /usr/bin/stockfish
```

For independent move-generation checks, `scripts/verify_perft.py` compares Alekhine with `python-chess`. Project formatting follows the workspace rules; do not run rustfmt.

The benchmark runner measures engine operations, search, UCI latency, and training. It requires a new output directory and retains raw samples and a report:
```sh
python3 scripts/benchmark.py --output build/benchmarks/baseline_001
```

Use `--baseline path/to/report.json` to compare another run. The benchmark harness links the current Rust library; pass the matching `--core_binary` when comparing an archived engine. Benchmark timings depend on the machine and do not measure playing strength.

## Matches

The paired-game runner needs Stockfish and `python-chess`. By default, it uses `build/release/alekhine`, the openings in `test/openings.epd`, and four opening pairs:
```sh
python3 scripts/play_stockfish.py \
    --stockfish /usr/bin/stockfish \
    --output build/matches/stockfish_001
```

It checks move legality, reverses colors for each opening, and retains PGN games, UCI transcripts, and binary fingerprints. It refuses an existing output directory. For larger paired matches, `scripts/match.py` prepares a Fastchess run with a candidate, baseline, opening suite, and sequential probability ratio test. Use `--dry-run` to inspect its command before starting a match.

## Limits and license

Alekhine currently supports standard chess on one CPU search worker. Chess960, tablebases, GPU inference, distributed training, and full Stockfish evaluation are not implemented. Tests and short matches have run on Linux x86-64; other platforms and long tournament play need separate verification.

Alekhine is written by Mayaz Rakib and licensed under the [GNU Affero General Public License, version 3 or later](LICENSE). Optional external tools, training data, and network weights retain their own licenses.
