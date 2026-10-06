# Alekhine

`alekhine` is a Rust engine for standard chess that speaks the Universal Chess Interface (UCI) protocol. It uses alpha-beta search with bounded policy-guided candidate ordering. Its built-in neural-network weights are untrained, and it has not shown a playing-strength advantage over Stockfish.

## Build and run

Alekhine requires Rust 1.85 or newer and uses only Rust's standard library.

```sh
cargo build --offline --locked --release
./target/release/alekhine
```

With no arguments, the binary reads UCI commands from standard input:

```text
uci
isready
position startpos moves e2e4 e7e5
go depth 4
```

Wait for `bestmove` before sending `quit`. Use `uci` for the option list. `Hash`, `MultiPV`, `Ponder`, `Policy Simulations`, and `EvalFile` are available; `Threads` currently accepts one worker. `perft`, `divide`, `bench`, and `version` are also available as commands.

For a development build and packaged archives:

```sh
python3 scripts/build.py
python3 scripts/package.py
```

The development binary is `build/release/alekhine`. Packaging writes archives to `dist/` and refuses to overwrite existing files.

## Networks and training

Alekhine uses legal move generation, reversible moves, incremental hashes, iterative deepening, quiescence, and selective pruning. It handles castling, en passant, promotion, repetition, the fifty-move rule, stalemate, and basic dead positions.

Its bootstrap evaluator is a quantized, untrained NNUE. It can also load supported Stockfish 18 large and small networks:

```text
setoption name EvalFile value /absolute/path/to/model.nnue
```

Set `EvalFile` to `<bootstrap>` to restore the bundled evaluator. No external engine source code or network weights are bundled.

The trainer accepts FEN positions, White-relative scores, optional outcomes and move probabilities, and game identifiers. It keeps whole games apart between training and validation, deduplicates positions, and refuses to overwrite outputs.

```sh
mkdir -p build/training
./build/release/alekhine train_network \
    build/training/positions.txt \
    build/training/model.nnue \
    10
```

Use `scripts/generate_training_data.py` with Stockfish and `python-chess` to label PGN games. `fine_tune_stockfish` supports compatible Stockfish 18 networks. Validation loss alone is not playing-strength evidence.

## Test and benchmark

```sh
python3 scripts/build.py --test
python3 scripts/benchmark.py --output build/benchmarks/baseline_001
```

Stockfish integration tests additionally require Stockfish 18 and `python-chess`; pass `--stockfish /usr/bin/stockfish`. `scripts/verify_perft.py` compares move generation with `python-chess`. Do not run `rustfmt`.

The benchmark runner retains raw samples and a report. Timings depend on the host and do not measure playing strength. `scripts/play_stockfish.py` runs paired games, while `scripts/match.py --dry-run` prepares a larger Fastchess match.

## Limits and licence

Alekhine currently supports standard chess on one CPU search worker. Chess960, tablebases, GPU inference, distributed training, and full Stockfish evaluation are not implemented. Linux x86-64 testing does not establish support for other platforms or long tournament play.

Alekhine is licensed under the [GNU Affero General Public License, version 3 or later](LICENSE). Optional external tools, training data, and network weights retain their own licences.
