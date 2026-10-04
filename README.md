# `alekhine`

[![Testing](https://img.shields.io/badge/Testing-187%20local%20tests-237c65)](#development-and-verification)
[![Rust](https://img.shields.io/badge/Rust-2024-237c65)](#installation-and-setup)
[![License: AGPL](https://img.shields.io/badge/License-AGPL--3.0--or--later-237c65)](LICENSE)

A Rust chess engine for search and neural evaluation research.

`alekhine` combines alpha-beta tactical search, policy-guided tree exploration, and incremental neural evaluation. Stockfish, Leela Chess Zero, and Reckless serve as algorithmic references; their source code is not incorporated. Optional Stockfish 18 networks can be loaded and their dense layers fine-tuned using Rust code. The research goal is to outperform Stockfish under controlled tournament conditions. The implementation includes CPU trainers and Stockfish match tooling, but it has no demonstrated strength advantage over Stockfish.

It features:
- Bitboards, compact moves, legal move generation, strict Forsyth-Edwards notation (FEN) parsing, and reversible move updates for standard chess.
- Castling, en passant, all four promotions, repetition history, fifty-move claims, stalemate, and basic dead-position detection.
- Incremental Zobrist and pawn hashes, incremental material, and history-sensitive transposition keys.
- Iterative deepening, negamax alpha-beta, principal variation search, quiescence, aspiration windows, and mate-distance pruning.
- Transposition bounds and aging, mate-score normalization, static exchange evaluation, killer moves, capture and continuation history, pawn correction history, late-move reductions, verified null-move pruning, reverse futility pruning, and bounded check and singular extensions.
- A bounded policy-guided predictor plus upper confidence bound applied to trees (PUCT) search with tactical leaf verification and empirical variance exploration. It receives at most 20 percent of the node or hard-time budget before alpha-beta verification.
- A version 2 quantized efficiently updatable neural network (NNUE) with king buckets, attack and defense features, incremental accumulators, squared clipped activations, a value head, an outcome head, and factorized move policy heads.
- An original Rust CPU trainer with joint value, outcome, and policy losses, AdamW optimization, game-separated validation, deduplication, integer export, and training provenance.
- Stockfish 18 large and small network loading, integer evaluation, and dense-layer fine-tuning with a frozen feature transformer and piece-square table (PSQT) weights.
- Configurable preference for checking and king-pressure moves within a bounded score margin, with mate scores taking priority.
- Versioned network import and export with dimension, weight, size, and checksum validation.
- Asynchronous Universal Chess Interface (UCI) support, legal fallback moves, hard cancellation, clock reserves, pondering, root restrictions, and multiple principal variations.
- Perft, perft divide, deterministic benchmarking, optional reference-engine verification, a paired Stockfish game runner, teacher-label generation, and a Fastchess match launcher with paired openings and sequential probability ratio testing (SPRT).
- Standard-library engine code, a reusable Cargo library, executable distribution archives, and a Cargo-driven development build script.

It is authored by Mayaz Rakib.

## Installation and Setup

Use Rust 1.85 or newer with Cargo and the Rust 2024 edition. The engine uses only Rust's standard library and has no third-party crate dependencies. No formatter is required, and project formatting follows the workspace rules. Do not run rustfmt.

From the project directory:
```sh
cargo build --offline --locked --release
cargo test --offline --locked --release
python3 test/uci_tests.py ./target/release/alekhine
python3 test/research_tests.py ./target/release/alekhine
python3 test/tool_tests.py
python3 test/integration_tests.py ./target/release/alekhine
python3 test/performance_tests.py ./target/release/alekhine
./target/release/alekhine
```

The Python build script retains the development executable paths and records build provenance:
```sh
python3 scripts/build.py --test
./build/release/alekhine
```

Set `CARGO` and `RUSTC` to select another toolchain. The script also discovers a standard installation in `~/.cargo/bin` when Cargo is absent from PATH. It uses only Python's standard library, treats compiler warnings as errors, and writes a build manifest containing compiler identity, flags, target architecture, revision, source digest, binary digest, build time, and test status.

Install the executable into a local prefix:
```sh
cargo install --offline --locked --path . --root dist/install
```

Rust consumers can depend on the `alekhine` library through a Cargo path dependency or a source package. The C++ headers, static library interface, CMake package, and CPack workflow have been replaced by Rust modules and Cargo. UCI commands, standalone engine commands, and version 1 network files remain compatible.

Create executable archives or a Cargo source package:
```sh
python3 scripts/package.py
cargo package --offline --locked --allow-dirty
```

The executable archives contain the binary, license, metadata, README, and build manifest. The packaging script verifies the binary digest and refuses to overwrite existing archives. Cargo's source package includes the Rust library, binary, core and hybrid tests, UCI and research tests, opening fixtures, and development scripts. Packaging does not publish the crate.

## Architecture

Both search systems use the same position representation, legal move generator, draw semantics, and neural feature encoding:
```text
UCI input
    |
    +--> Position and move history
    |
    +--> Search control and time budget
             |
             +--> Policy/value PUCT tree
             |        |
             |        +--> Root visit counts and policy priors
             |
             +--> Iterative alpha-beta and PVS
                      |
                      +--> Root ordering and MultiPV
                      +--> Quiescence and selective reductions
                      +--> Transposition table and search history
                      +--> Incremental neural evaluation
                               |
                               +--> Bitboards and reversible piece changes
```

PUCT explores legal moves using policy priors and side-to-move values. It backs up values with alternating signs and uses visit counts to order root candidates. Leaves receive bounded alpha-beta verification. An optional empirical branch-variance bonus encourages exploration; it is not calibrated model uncertainty. PUCT and alpha-beta share the node budget, clock, and stop flag, with at most 20 percent allocated to PUCT. Storage has explicit node and edge limits. Alpha-beta uses bound searches to reject clearly inferior root candidates and fully searches competitive candidates. MultiPV searches each root candidate separately.

Version 2 NNUE has 6,288 inputs: eight king buckets of 768 piece-square features, plus 144 attack and defense interaction classes. Two color-normalized perspectives share a 128-unit transformation. Their 256 activations feed value, win/draw/loss, origin, destination, and promotion heads. Squared clipped rectified linear unit (SCReLU) activations clip each hidden activation and square it. Integer inference uses signed 16-bit weights, 32-bit accumulators, and 64-bit output arithmetic. Make/unmake records affected squares; accumulator updates merge changed sparse features, including sliding attack changes and king-bucket transitions.

The default version 1 bootstrap model retains 768 inputs, 32 hidden units, and handcrafted, untrained weights for compatibility. A version 2 model can train value, policy, and outcome heads independently. Missing learned heads fall back to bootstrap policy or a score-based outcome approximation. Outcome outputs are uncalibrated until independently evaluated. Stockfish 18 large and small networks are supported as a separate value-evaluation backend. Other Stockfish architectures, Lc0 weights, and Reckless weights are unsupported.

This is a CPU research implementation. Transformer policy networks, batched GPU inference, distributed training, and production-scale training are not implemented. No Stockfish, Lc0, or Reckless source code or weights are bundled.

The main modules are:

| Module | Responsibility |
|---|---|
| [position.rs](src/position.rs) | Board state, legal moves, hashing, history, draw detection, and perft. |
| [network.rs](src/network.rs) | Quantized weights, incremental accumulators, inference, and network files. |
| [neural.rs](src/neural.rs) | Version 2 features, integer inference, independent heads, and serialization. |
| [search.rs](src/search.rs) | PUCT, alpha-beta search, principal variations, limits, and time allocation. |
| [tactical.rs](src/tactical.rs) | Static exchange estimates and king-pressure move priorities. |
| [training.rs](src/training.rs) | Dataset validation, backpropagation, optimization, and quantized export. |
| [stockfish_network.rs](src/stockfish_network.rs) | Stockfish 18 decoding, feature encoding, integer evaluation, and export. |
| [stockfish_training.rs](src/stockfish_training.rs) | Dense-layer fine-tuning, quantized validation, and provenance. |
| [transposition_table.rs](src/transposition_table.rs) | Bounded storage, replacement, score bounds, aging, and mate normalization. |
| [uci.rs](src/uci.rs) | Command validation, search worker, synchronized output, and cancellation. |
| [scripts/](scripts/) | The scripts provide local builds, reference verification, and external match orchestration. |

Move generation currently filters pseudo-legal candidates through make/unmake. Sliding attacks use portable ray traversal. These are correctness baselines for profiling and later replacement with specialized evasions, attack tables, staged move picking, and architecture-specific instructions.

## Configuration

Existing UCI option names retain their conventional spelling. New project-controlled options and internal identifiers use the workspace naming rules.

| Option | Default | Purpose |
|---|---|---|
| `Hash` | `16` | Sets transposition storage from 1 to 512 MiB. |
| `Clear Hash` | Button | Clears stored transpositions. |
| `Threads` | `1` | Accepts one search worker. |
| `MultiPV` | `1` | Reports up to 16 root variations. |
| `Ponder` | `false` | Advertises pondering; the GUI starts it with `go ponder`. |
| `Move Overhead` | `20` | Reserves milliseconds from the clock or move time. |
| `Policy Simulations` | `128` | Sets the PUCT budget from 0 to 4,096; zero disables PUCT. |
| `policy_verification_depth_plies` | `1` | Sets tactical leaf verification from 0 to 3 plies. |
| `policy_uncertainty_percent` | `25` | Sets the variance exploration bonus from 0 to 100 percent. |
| `aggression_margin_centipawns` | `15` | Prefers king-pressure moves within 0 to 100 centipawns; zero disables the preference. |
| `is_selective_search_enabled` | `true` | Enables selective pruning, reductions, extensions, and correction history. |
| `EvalFile` | `<bootstrap>` | Selects a network file or restores the untrained built-in model. |

The command stream remains responsive during search. Position changes, option changes, and new-game commands stop and join the current worker before mutating its resources. Malformed commands report UCI-compatible `info string` messages. Position and network replacement are transactional, so rejected input preserves the previous valid state. Standard output belongs exclusively to UCI when the engine starts without arguments.

The engine bounds commands to 65,536 bytes, FEN input to 256 bytes, network files to two MiB, search recursion to 128 plies, and position history to 4,096 entries. UCI reserves history capacity for search. Tablebases and Chess960 are not supported in this version, and no unsupported options are advertised.

## Commands and Analysis

Start the binary with no arguments, then send UCI commands through a GUI, tournament manager, or terminal:
```text
uci
isready
position startpos moves e2e4 e7e5
setoption name MultiPV value 3
go depth 4
```

Wait for `bestmove` before entering `quit`. Other supported searches include `go nodes 10000`, `go movetime 1000`, `go wtime 60000 btime 60000 winc 100 binc 100`, `go infinite`, and `go searchmoves g1f3 depth 4`. Stop a running search with `stop`. For pondering, use `go ponder`, then `ponderhit` when the expected opponent move arrives.

Analysis reports depth, selective depth, side-to-move score, nodes, nodes per second, elapsed milliseconds, hash fullness, and legal principal variations. Mate scores use a separate reserved range and UCI mate notation. Tablebase hits are always zero in this version. `bestmove 0000` identifies a root with no legal moves.

Standalone commands are available for development:

| Command | Purpose |
|---|---|
| `./build/release/alekhine version` | The command reports identity, copyright, license, compiler, revision, and network fingerprint. |
| `./build/release/alekhine perft 5` | The command counts legal leaf nodes from the initial position. |
| `./build/release/alekhine divide 3 "FEN"` | The command reports per-root-move leaf counts for a supplied position. |
| `./build/release/alekhine bench 4` | The command searches a fixed suite and reports nodes, time, throughput, and a deterministic checksum. |
| `./build/release/alekhine export_network path.network` | The command exports the untrained bootstrap model in the current network format. |
| `./build/release/alekhine train_network positions.txt model.nnue 10 --batch_size 256 --learning_rate 0.001 --seed 20261004` | The command trains and exports a version 2 model and a training manifest. |
| `./build/release/alekhine fine_tune_stockfish source.nnue positions.txt tuned.nnue 20 --learning_rate 0.0001` | The command fine-tunes a supported Stockfish 18 network and exports a compatible network and training manifest. |
| `./build/release/alekhine inspect_stockfish_network model.nnue "FEN"` | The command reports architecture, fingerprint, material bucket, and integer scores for all eight buckets as JSON. The initial position is used when FEN is omitted. |

Version 1 uses the existing text representation. Version 2 uses explicit little-endian binary dimensions, scales, independent trained-head markers, quantized tensors, and an FNV checksum. The checksum detects accidental corruption; it is not a cryptographic authenticity check. Training markers record exported capabilities and do not prove strength. Both formats validate architecture, size, weights, and checksum before replacing the current network.

## Development and Verification

The suite contains 187 tests without external chess dependencies, plus 13 explicitly selected Stockfish and python-chess integration tests. Tests use isolated temporary directories, deterministic seeds, bounded subprocess deadlines, and protocol synchronization. They do not use rustfmt.

| Suite | Count | Coverage |
|---|---|---|
| Rust unit tests in `src/` | 46 | Move encoding, coordinate parsing, fingerprints, transposition replacement, NNUE integer arithmetic and file validation, independent heads, training losses and gradients, tactical attacks, clock control, signed network compression, dense-layer gradients, and padding preservation. |
| [core_tests.rs](tests/core_tests.rs) | 27 | Canonical perft positions, FEN parsing, reversible state, draw rules, hashing, transpositions, search results, limits, and reproducibility. |
| [hybrid_tests.rs](tests/hybrid_tests.rs) | 13 | Neural accumulator updates, special moves, symmetry, policy batching, model loading, null moves, static exchanges, root verification, and small node budgets. |
| [state_tests.rs](tests/state_tests.rs) | 13 | Castling rights, counters, en passant and pawn hashes, tactical move generation, full stack restoration, history capacity, cancellation, ablations, null-subtree repetition, and mate priority. |
| [training_tests.rs](tests/training_tests.rs) | 7 | Deterministic training/export/load, validation-label isolation, deduplication, option and record limits, malformed input, and output preservation. |
| [uci_tests.py](test/uci_tests.py) | 19 | Real UCI subprocess identity, search, clocks, MultiPV, root restrictions, pondering, stopping, lifecycle transitions, invalid input, and options. |
| [research_tests.py](test/research_tests.py) | 4 | Option/report parsing, handshake rejection, and training/export/UCI loading. |
| [tool_tests.py](test/tool_tests.py) | 15 | Client timeouts, crash diagnostics, oversized output, command injection rejection, report parsing, match configuration, and archive integrity. |
| [integration_tests.py](test/integration_tests.py), engine suite | 15 | Compiled CLI commands, training manifests, corrupt-model recovery, active-search model replacement, trained special moves and MultiPV, EOF shutdown, invalid UTF-8, and history/input limits. |
| [integration_tests.py](test/integration_tests.py), Stockfish suite | 8 | Actual clocked pairs, teacher/trainer/model/match workflows, legal PGNs and provenance, illegal-move and time forfeits, crash incompleteness, output preservation, and independent perft across 128 positions. |
| [stockfish_training_tests.py](test/stockfish_training_tests.py), format suite | 9 | Synthetic files, dense weight changes, frozen-byte preservation, deterministic training, quantized baseline selection, validation isolation, deduplication, input rejection, and UCI loading. |
| [stockfish_training_tests.py](test/stockfish_training_tests.py), Stockfish suite | 5 | Actual large and small networks, byte-identical exports, evaluation against Stockfish across legal positions and all material buckets, and modified network loading and search in both engines. |
| [performance_tests.py](test/performance_tests.py) | 19 | Statistics, compatible baselines, regression gates, deterministic engine workloads, search budgets, NNUE accumulators, synthetic datasets, complete artifacts, output preservation, failure logs, and timeout cleanup. |

The protocol fixture deliberately injects failures that would be impractical to reproduce reliably with a normal engine. Its configurable delay tests clock overruns; other lifecycle tests synchronize on protocol output. Real Stockfish and python-chess provide independent chess semantics in the optional suite. These are bounded functional checks, not exhaustive path coverage, fuzzing, long tournament soak tests, or strength measurements.

| Command | Purpose |
|---|---|
| `python3 scripts/build.py --test` | The command builds the portable release engine and runs all 187 tests without external chess dependencies. |
| `python3 scripts/build.py --mode debug --test` | The command enables debug information, bounds checks, and integer overflow checks. |
| `python3 scripts/build.py --test --stockfish /usr/bin/stockfish` | The command runs all 200 tests, including real Stockfish and python-chess workflows. Use Python from an environment with python-chess installed. Network checks require Stockfish 18. |
| `python3 test/integration_tests.py ./build/release/alekhine --stockfish /usr/bin/stockfish` | The command runs the 23 engine and Stockfish integration tests against an existing build. |
| `python3 scripts/build.py --native --lto --test` | The command enables host-specific instructions and link time optimization; the binary may require the build host's CPU features. |
| `cargo test --offline --locked --release` | The command runs the Rust core tests. |
| `cargo clippy --offline --all-targets -- -D warnings` | The command checks Rust code and tests without formatting them. |
| `python3 scripts/package.py` | The command creates verified executable archives from the selected build. |
| `python3 scripts/verify_perft.py ./build/release/alekhine --positions 1000` | The command checks legal moves and per-root counts against `python-chess`; this optional tool requires that development dependency. |

The Rust migration passed an independent comparison for 1,000 deterministic legal positions at depth 2, covering 1,046,903 leaf nodes with seed `20261004`. All 45 tests passed in release and debug builds. The Rust and saved C++ binaries produced identical depth-4 benchmark node counts and checksums, and their bootstrap network exports were byte-for-byte identical. These are finite correctness and compatibility checks; they do not establish playing strength. Linux x86-64 is the verification platform. Windows, macOS, ARM64, extended tournament runs, and playing strength require separate verification.

The previous C++ sanitizer mode has been removed. Rust debug builds check bounds and integer overflow, and the crate forbids unsafe code. These checks do not establish sanitizer coverage. Clippy's preference for implicit returns is disabled to retain the project's explicit control-flow style; other warnings remain errors during lint verification.

For reproducible search comparisons, keep the binary, network fingerprint, initial FEN and history, cleared hash, thread count, policy budget, and node limit fixed. Benchmark throughput is hardware-dependent. Correctness tests and deterministic checksums do not measure Elo.

## Performance Testing and Benchmarking

Run the repeatable performance suite against a release build:
```sh
python3 scripts/build.py
python3 scripts/benchmark.py --output ./build/benchmarks/baseline_001
```

Use `--network /absolute/path/to/model.nnue` to measure both the bootstrap evaluator and a loaded network. The runner uses only the standard library, discovers Cargo through the build script, and compiles a separate release harness in `build/benchmark/cargo`. Compilation is outside measured workloads. The existing `alekhine bench` command remains available.

| Suite | Measured behavior |
|---|---|
| `micro` | The harness measures FEN parsing and serialization, full and tactical legal moves, legal-move existence, make/unmake, full and incremental accumulators, neural feature encoding, value evaluation, batched policy evaluation, static exchanges, transposition stores and probes, and three canonical perft workloads. |
| `search` | The harness measures nine opening, castling, endgame, en passant, promotion, and check-evasion positions under fixed-node and fixed-depth budgets, with cleared and primed transpositions. Profiles include the default hybrid, disabled policy exploration, disabled policy and selective search, MultiPV 3, and 1, 16, and 64 MiB hash configurations. |
| `latency` | Fresh engine processes measure startup and configured-model readiness, idle and active readiness, stop response, ponderhit response, and a 50-millisecond search with zero move overhead. |
| `training` | Fresh processes train all three heads on deterministic, deduplicated, legal synthetic corpora of 32, 256, and 2,048 positions. Measurements include loading, encoding, optimization, validation, quantization, and export. Synthetic labels are workload fixtures, not chess-quality targets. |

The default suite is `all`, with five measured samples and one separately retained first pass per case. Microbenchmarks calibrate bounded loop counts toward 15 milliseconds, then execute two additional unmeasured warmups. Each operation rotates through nine positions. Search timing includes constructing search state and the complete search, with table allocation, clearing, and optional priming outside the timer. Search reports retain the actual final node count, completed depth, score, and legal best move. Depth-limited workloads can visit different numbers of nodes when search profiles or implementations change. A warm table is primed once before each measured search; it does not retain an unlimited history across samples. Neither first-pass nor cleared-table measurements imply flushed CPU or disk caches.

The output directory must be new. It retains raw JSONL samples, protocol and subprocess logs, model fingerprints, binary and network SHA-256 digests, compiler and available build provenance, environment and CPU affinity, workload identities, generated datasets, trained exports, a JSON report, and CSV summaries. Statistics include median, mean, minimum, maximum, interpolated p95, sample standard deviation, and coefficient of variation. Five samples provide a small descriptive distribution, not a confidence interval or a significance test. Training throughput is corpus size times configured epochs divided by total process duration, rather than an isolated backpropagation rate.

On Linux with GNU Time, whole subprocess reports include CPU time, peak resident memory, page faults, and context switches. UCI cases use Linux process CPU ticks and the process memory high-water mark, which includes earlier operations in that process. Unsupported resource measurements are null. The suite does not measure allocations, hardware counters, power, GPU performance, or playing strength. WSL, host load, scheduling, thermal state, and CPU frequency can affect timings. Run sequentially on an idle machine, retain affinity settings, and repeat noisy cases.

Select `--suite micro`, `search`, `latency`, or `training` for a focused run. Bounds and workload controls include `--samples`, `--minimum_ms`, `--nodes`, `--depth`, `--training_records`, `--epochs`, and `--batch_size`. Individual subprocesses have a 600-second deadline. Failures retain a failed report and available logs; timeout cleanup terminates the measured process group on POSIX systems. Functional tests exercise small workloads without host-sensitive absolute timing assertions.

Compare a candidate to a retained baseline with identical workload settings:
```sh
python3 scripts/benchmark.py \
    --output ./build/benchmarks/candidate_001 \
    --baseline ./build/benchmarks/baseline_001/report.json \
    --maximum_regression_percent 10
```

The runner rejects incomplete baselines, changed suites, different networks, mismatched workloads, and different environments. `--allow_environment_change` permits an explicitly exploratory environment comparison. Exit status 2 means at least one median duration exceeded the configured regression threshold; measurements remain complete and reviewable. This gate checks observed duration ratios, not statistical significance. A threshold should account for the retained variation and repeated independent runs.

The Rust harness measures its own linked engine library, while UCI and training use `--engine`. Preserve the compiled harness listed in each report and pass `--core_binary` when comparing an archived engine implementation. Supplying an older `--engine` alone does not change the library linked into a freshly built harness. Run the standalone harness through Cargo with `cargo bench --offline --locked --bench performance -- --suite micro` when raw JSONL is sufficient.

## Engine Matches and Research

The paired-game runner uses an installed Stockfish executable as an opponent. It checks move legality, enforces wall-clock budgets, reverses colors for each opening, and retains PGN, UCI transcripts, binary and network SHA-256 digests, options, and results. The Python match and teacher scripts require `python-chess`; install it in a development virtual environment. The engine and Rust trainer do not require Python dependencies.

Run a paired match with:
```sh
python3 scripts/play_stockfish.py \
    --candidate ./build/release/alekhine \
    --stockfish /usr/bin/stockfish \
    --openings ./test/openings.epd \
    --output ./build/matches/stockfish_001 \
    --pairs 4 \
    --clock_milliseconds 30000 \
    --increment_milliseconds 100
```

Supply `--candidate_option EvalFile=/absolute/path/to/model.nnue` to test a learned network. Both engines default to one thread and 16 MiB hash. Repeated `--candidate_option` and `--stockfish_option` arguments configure advertised UCI options. The runner refuses existing output directories. Games reaching the configurable ply limit are explicitly adjudicated, and forfeits are recorded separately from normal game endings. This short runner provides functional live-game testing, not a statistical strength certificate.

Use [Fastchess](https://github.com/Disservin/fastchess) for larger matches, paired statistics, and sequential probability ratio testing. Install it separately, provide candidate and baseline configurations, and supply an attributed PGN or EPD opening suite:
```sh
python3 scripts/match.py \
    --candidate ./build/release/alekhine \
    --baseline /absolute/path/to/baseline \
    --openings /absolute/path/to/openings.epd \
    --output ./build/matches/candidate_001 \
    --time-control 10+0.1 \
    --rounds 10000 \
    --concurrency 1 \
    --dry-run
```

Remove `--dry-run` to start the prepared match. The launcher refuses an existing output directory and records binary and opening-suite SHA-256 digests, the command, and the start time. Fastchess owns paired statistics, SPRT, logs, and resumable match configuration. The launcher's default hypotheses use logistic Elo thresholds of 0 and 5 with 0.05 error probabilities. They are configurable research parameters. Match completion alone does not mean SPRT accepted the candidate. The launcher was verified through command preparation and a dry run; a Fastchess match has not been run for this initial version.

Repeated `--candidate_option` and `--baseline_option` arguments support network and search ablations. The same binary can appear on both sides when its option configurations differ. Comparing against Stockfish requires pinned binaries, equal hardware and memory budgets, documented neural weights, balanced openings, long enough runs, and retained results. The live verification opponent in this workspace is Stockfish 18; no claim about the latest release is made.

## Neural Training

Generate labels from complete PGN games using Stockfish as a teacher. The generator samples quiet, nonterminal positions, retains exact MultiPV scores and legal policy targets, and deduplicates positions. Teacher code and weights remain in the separate teacher process. Labels use White-relative centipawns and game outcomes. Missing or forfeited outcomes remain unknown.

Generate a dataset and train a model with:
```sh
python3 scripts/generate_training_data.py \
    --pgn ./build/matches/stockfish_001/games.pgn \
    --teacher /usr/bin/stockfish \
    --output ./build/training/positions.txt \
    --nodes 3000 \
    --maximum_records 100000 \
    --sample_interval 2
./build/release/alekhine train_network \
    ./build/training/positions.txt \
    ./build/training/model.nnue \
    10 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --seed 20261004
```

Create the training directory before writing outputs. Both tools refuse existing outputs. The dataset and model receive provenance manifests. Training records have this form:
```text
FEN | White score in centipawns | White result or ? | move:probability ... or ? | game_id
```

Results are `0`, `0.5`, or `1`. Policy probabilities are nonnegative and normalized by the trainer. Duplicate positions are removed using the first four FEN fields before splitting. A deterministic hash assigns complete game groups to approximately 20 percent validation; both partitions must be nonempty. Value targets mix sigmoid teacher scores with known game outcomes. Outcome and legal-move policy heads use cross-entropy losses. Unknown outcomes do not train the outcome head.

The trainer caches records in memory, defaults to a maximum of 100,000 unique positions, and supports `--maximum_record_count` up to 1,000,000. It selects the best validation epoch, quantizes weights, and measures score differences on at most 256 validation records. It has no GPU backend, distributed training, checkpoint resume, or quantization-aware training. For strength experiments, use a much larger, diverse corpus, higher-quality teacher searches, independently held-out openings, and long paired matches. Validation loss alone does not establish improved play.

The verified smoke model used 100 positions from earlier Stockfish games, with 78 training and 22 validation records. Ten epochs selected epoch 4, reducing validation loss from 5.166842 to 4.957920. Maximum sampled quantization error was 4.83 centipawns. The model then lost eight of eight paired games against Stockfish 18 at 5 seconds plus 50 milliseconds per move, all by checkmate. The final bootstrap evaluator also lost all eight games under the same conditions. Both series completed without forfeits. Results are retained locally in `build/matches/original_hybrid_trained_20261004` and `build/matches/original_hybrid_verified_bootstrap_20261004`. These games verify actual play and the exported model; the small dataset and reused opening families do not support an Elo estimate or a strength advantage.

### Fine-Tuning Stockfish Weights

The separate `fine_tune_stockfish` command starts directly from a Stockfish 18 `.nnue` file. It supports the large 1,024-channel HalfKAv2_hm plus FullThreats network and the small 128-channel HalfKAv2_hm network. Version `0x7af32f20` and architecture hashes `0xec102ef2` and `0x1c103c92` identify the supported formats. Other architectures are rejected. Compatibility is pinned to the [Stockfish 18 architecture](https://github.com/official-stockfish/Stockfish/blob/sf_18/src/nnue/nnue_architecture.h).

Export weights from an installed Stockfish 18 executable into unused paths, then fine-tune either network using the existing dataset format:
```sh
mkdir -p ./build/stockfish
/usr/bin/stockfish <<'EOF'
export_net ./build/stockfish/source_big.nnue ./build/stockfish/source_small.nnue
quit
EOF
./build/release/alekhine fine_tune_stockfish \
    ./build/stockfish/source_big.nnue \
    ./build/training/positions.txt \
    ./build/stockfish/tuned_big.nnue \
    20 \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --seed 20261004
```

Stockfish's `export_net` may overwrite its destinations, so choose unused paths. Alekhine's trainer refuses existing model or manifest outputs. Official Stockfish network weights are [CC0](https://github.com/official-stockfish/networks/blob/master/README.md). Engine source is not copied or required for training, and weights are not bundled in the package.

Training freezes the entire feature transformer and PSQT weights, preserving their serialized bytes exactly. AdamW updates all three dense affine layers in material buckets represented by training records, excluding unused padding. The large architecture has up to 139,400 trainable parameters across eight buckets; the small architecture has up to 24,712. Buckets absent from training remain unchanged. This implementation does not fine-tune the sparse transformer or threat embeddings.

The trainer reuses record validation, position deduplication, game-group splitting, and side-relative mixed score/outcome targets. Policy labels are validated but do not enter the Stockfish value loss. Cached transformer activations feed floating-point backpropagation. Each epoch is quantized and evaluated with exact integer inference on validation records. The best quantized epoch is retained, including the unchanged source network as epoch zero. Floating-point and integer scores can differ materially; the manifest reports their maximum sampled difference. This is not quantization-aware training, GPU training, or full-network training.

The output retains Stockfish's description, format, architecture, and frozen prefix. A sibling `.training.json` records source and output FNV fingerprints, dataset fingerprint, counts, seed, hyperparameters, frozen layers, changed parameters, losses, and sampled quantization error. These fingerprints are noncryptographic. Fine-tuning an earlier exported model continues from its weights with a fresh optimizer; optimizer state is not resumed. The shared `--maximum_record_count` option bounds the cached dataset.

Load the result in Alekhine using UCI:
```text
uci
setoption name EvalFile value /absolute/path/to/tuned_big.nnue
isready
position startpos
go nodes 10000
```

For Stockfish 18, use `EvalFile` for a large network or `EvalFileSmall` for a small network. Alekhine loads one selected network, recomputes its full features for each evaluation, and converts its raw PSQT plus positional score to centipawns using the pinned Stockfish 18 normalization. Incremental Stockfish accumulators and SIMD are not implemented. Alekhine retains its own search, bootstrap policy priors, and score-based outcome approximation; its search evaluation does not reproduce Stockfish's optimism, complexity scaling, or rule-50 damping.

The local large-network smoke run used the existing 100 teacher positions, split into 78 training and 22 validation records. Twenty epochs changed 168 dense parameters and reduced quantized validation loss from 0.537171 to 0.536950. Maximum sampled floating-point versus integer score difference was 61.37 centipawns. The local small-network run did not improve validation, so its export retained the source weights. These results demonstrate compatible fine-tuning and conservative model selection; they do not establish improved tournament strength. Artifacts are retained locally in `build/stockfish/fine_tuned_big.nnue` and its training manifest.

The fine-tuned large model completed a color-reversed pair against Stockfish 18 at 5 seconds plus 50 milliseconds per move. Both games ended in checkmate losses without forfeits. The pair is retained in `build/matches/stockfish_fine_tuned_20261004`. A three-sample benchmark of complete fresh-process fine-tuning on 100 positions for 20 epochs measured median durations of 0.342 seconds for the large network and 0.023 seconds for the small network. These smoke timings include file loading, feature caching, training, validation, and export; they do not estimate full-transformer or large-corpus training time. Training and UCI latency reports are retained in `build/benchmarks/stockfish_fine_tuning_20261004` and `build/benchmarks/stockfish_fine_tuned_latency_20261004`.

## Remaining Research Work

The remaining implementation and evaluation sequence is:
1. Expand core verification with larger perft suites, pathological positions, state validation, fuzzing, and long-running UCI tests.
2. Profile move generation, quiescence, root search, neural features, and hash access. Add specialized move generators, attack tables, and a staged move picker.
3. Measure the implemented pruning, extensions, history, aggression preference, and variance exploration independently through paired ablations. Tune parameters using retained tournament results.
4. Add race-free shared transpositions, bounded parallel search, better time stability metrics, and tournament soak testing.
5. Integrate Syzygy WDL and DTZ with explicit fifty-move semantics. Add Chess960 as a separately verified rules mode.
6. Scale licensed position generation and self-play, improve independent validation, add quantization-aware training, and establish network promotion through match evidence.
7. Expand the hybrid with a stronger policy/value model, batched inference, GPU backends, and calibrated scheduling between tactical verification and policy exploration.
8. Add architecture-specific SIMD, topology-aware memory, profile-guided builds, parameter tuning, and an established distributed testing system such as OpenBench or Fishtest.

These are research milestones, not shipped capabilities or promises of a particular rating.

## Licensing

Original project code uses the [GNU Affero General Public License, version 3 or later](LICENSE), with copyright held by Mayaz Rakib. Package identity, description, author, license, and Rust edition are recorded in [Cargo.toml](Cargo.toml) and [project.toml](project.toml).

[AGENTS.md](AGENTS.md) sets package-specific formatting rules. The workspace rules also apply to this project. Stockfish, Leela Chess Zero, Fastchess, optional development tools, training data, and external networks retain their respective licenses; none are relicensed through this project.
