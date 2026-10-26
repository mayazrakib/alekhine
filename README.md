# `alekhine`

![Version](https://img.shields.io/badge/Version-0.1.0-737373?style=flat-square)
![License](https://img.shields.io/badge/License-AGPL-3.0-or-later-737373?style=flat-square)
![Languages](https://img.shields.io/badge/Languages-Rust%2C%20Python-737373?style=flat-square)
![Frameworks](https://img.shields.io/badge/Frameworks-Cargo%2C%20python-chess-737373?style=flat-square)
![Compatibility](https://img.shields.io/badge/Compatibility-Not%20verified-737373?style=flat-square)
![Build](https://img.shields.io/badge/Build-Not%20run%20remotely-737373?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-Rust%20and%20Python%20suites%20passing-737373?style=flat-square)
![Vulnerabilities](https://img.shields.io/badge/Vulnerabilities-0%20known%20dependency%20findings-737373?style=flat-square)

## 1. Introduction

Alekhine is a standard-chess UCI research engine with alpha-beta search and bounded policy-guided ordering. Its bootstrap neural weights are untrained; no playing-strength advantage over Stockfish has been demonstrated.

## 2. Usage

Run `uv sync --locked`, then `uv run python tools/scripts/build.py` and `./build/release/alekhine`. Rust 1.98.0 is pinned; Rust 1.85 remains the declared minimum.

Send `uci`, `isready`, `position startpos moves e2e4 e7e5`, and `go depth 4` over standard input. Wait for `bestmove` before `quit`. `Hash`, `MultiPV`, `Ponder`, `Policy Simulations`, and `EvalFile` are supported; `Threads` accepts one worker.

`uv run python tools/scripts/package.py` creates archives in `dist/` and refuses to overwrite them.

## 3. Features

Legal move generation, reversible state, incremental hashes, iterative deepening, quiescence, pruning, castling, en passant, promotion, repetition, the fifty-move rule, stalemate, and basic dead positions are implemented.

`EvalFile` accepts supported external Stockfish 18 large/small networks; `<bootstrap>` restores the bundled evaluator. Training separates whole games across training/validation, deduplicates positions, and refuses output replacement. No external engine source or network weights are bundled.

## 4. Architecture

`src/` separates chess state, search, evaluation, UCI protocol, and training. `tools/scripts/` owns build, packaging, matches, and labeling. `tests/` contains Rust/Python tests and `tests/perf/` contains profiling.

Run `uv run python tools/scripts/build.py --test`; pass `--stockfish /usr/bin/stockfish` for real Stockfish 18 integration. `tools/scripts/verify_perft.py` compares move generation with python-chess. Do not run rustfmt. Use `uv run python tests/perf/benchmark.py --output build/benchmarks/baseline_001` for retained benchmark samples. Timings and validation loss do not establish playing strength.

## 5. Dependencies

| Dependency | Version | Use |
|---|---|---|
| `python-chess` | `1.999` | runtime |
| `Rust standard library` | `1.98.0` | engine runtime and build |

## 6. Roadmap

- [x] Single-worker standard chess, UCI, and training tooling.
- [ ] Chess960, tablebases, GPU inference, and distributed training are not implemented.

## 7. Contributing

Read `AGENTS.md` and the applicable skills in `.ai/skills/` before contributing. Set up the development environment with `uv sync --locked`. Follow the repository conventions, preserve supported behavior, and add relevant tests for behavior changes. Run `uv run --locked python tools/scripts/build.py --test` and `uv run --locked python tools/scripts/build.py` before opening a pull request. Describe the change, its purpose, and the validation results in the pull request.

## 8. License

`alekhine` is licensed under the AGPL-3.0-or-later license. See LICENSE for the complete license text.

See `ACKNOWLEDGEMENTS.md` for third-party notices and attributions.
