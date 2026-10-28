# `alekhine`

![Version](https://img.shields.io/badge/Version-0.1.0-737373?style=flat-square)
![License](https://img.shields.io/badge/License-AGPL--3.0--or--later-737373?style=flat-square)
![Languages](https://img.shields.io/badge/Languages-Rust%2C%20Python-737373?style=flat-square)
![Frameworks](https://img.shields.io/badge/Frameworks-Cargo%2C%20python--chess-737373?style=flat-square)
![Compatibility](https://img.shields.io/badge/Compatibility-Not%20verified-737373?style=flat-square)
![Build](https://img.shields.io/badge/Build-Not%20verified-737373?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-Rust%20and%20Python%20suites%20passing-737373?style=flat-square)
![Vulnerabilities](https://img.shields.io/badge/Vulnerabilities-0%20known%20dependency%20findings-737373?style=flat-square)

## 1. Introduction

Alekhine is a UCI research engine for standard chess. It combines alpha-beta search with bounded, policy-guided move ordering. Its bootstrap neural weights are untrained, and it hasn't shown any playing-strength advantage over Stockfish.

## 2. Usage

Install Rust 1.98.0, Python 3.14.7, uv 0.12.10, and Task 3.54.0. Rust 1.85 is the declared minimum. Then build and start the engine:

```sh
uv sync --locked
uv run python tools/scripts/build.py
./build/release/alekhine
```

Send UCI commands over standard input, and wait for `bestmove` before sending `quit`:

```text
uci
isready
position startpos moves e2e4 e7e5
go depth 4
```

The engine supports the `Hash`, `MultiPV`, `Ponder`, `Policy Simulations`, and `EvalFile` options. `Threads` accepts one worker.

`uv run python tools/scripts/package.py` creates archives in `dist/` and refuses to overwrite existing ones.

## 3. Features

- Legal move generation, including castling, en passant, and promotion.
- Reversible state and incremental hashing.
- Iterative deepening, quiescence search, and pruning.
- Draw detection for repetition, the fifty-move rule, stalemate, and basic dead positions.
- `EvalFile` support for external Stockfish 18 large and small networks; `<bootstrap>` restores the bundled evaluator.
- Training that splits whole games between training and validation, deduplicates positions, and refuses to replace existing output.

No external engine source or network weights are bundled.

## 4. Architecture

`src/` separates chess state, search, evaluation, the UCI protocol, and training. `tools/scripts/` owns building, packaging, matches, and labeling. `tests/` holds the Rust and Python tests, and `tests/perf/` holds profiling.

### Verification

`uv run python tools/scripts/build.py --test` builds and tests the engine. Add `--stockfish /usr/bin/stockfish` to include real Stockfish 18 integration tests. `tools/scripts/verify_perft.py` compares move generation with python-chess. `uv run python tests/perf/benchmark.py --output build/benchmarks/baseline_001` records retained benchmark samples. Timings and validation loss don't establish playing strength.

Don't run `rustfmt` on this repository.

## 5. Dependencies

| Dependency | Version | Use |
|---|---|---|
| `python-chess` | 1.999 | Python tools: training and move-generation verification |
| Rust standard library | 1.98.0 | Engine runtime and build; no external crates are declared |

## 6. Roadmap

- [x] Single-worker standard chess, UCI, and training tooling.
- [ ] Chess960, tablebases, GPU inference, and distributed training aren't implemented.

## 7. Contributing

Read `AGENTS.md` and the applicable skills in `.ai/skills/` before contributing. Set up the development environment with `uv sync --locked`. Follow the repository conventions, preserve supported behavior, and add relevant tests for behavior changes. Run `uv run --locked python tools/scripts/build.py --test` and `uv run --locked python tools/scripts/build.py` before opening a pull request. Describe the change, its purpose, and the validation results in the pull request.

## 8. License

`alekhine` is licensed under the AGPL-3.0-or-later license. See LICENSE for the complete license text.
