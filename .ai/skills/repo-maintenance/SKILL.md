---
name: repo-maintenance
description: "Maintain the alekhine repository and verify its supported development workflows."
---

## 1. Scope

Use when maintaining this repository or its independently configured projects.

## 2. Procedure

- Initialize with `uv sync --locked`; use `task test` to build and execute Rust, UCI, training, tool, and performance regression tests.
- Preserve UCI option spelling, command output contracts, network formats, and external NNUE files. The MIT license covers original repository code; external engines and networks retain their own terms.
- `build.rs` is Cargo's required build-script entry point. Keep the Rust source module layout and explicit package inclusion list synchronized with any file moves.
