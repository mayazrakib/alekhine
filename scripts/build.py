import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent

def run_command(
    arguments: list[str],
    environment: dict[str, str],
) -> None:
    subprocess.run(
        arguments,
        cwd=PROJECT_DIRECTORY,
        env=environment,
        check=True,
    )

def resolve_tool(command_name: str) -> list[str]:
    command = shlex.split(os.environ.get(
        command_name.upper(),
        command_name,
    ))

    if not command:
        raise ValueError(f"Failed to resolve {command_name} because its configured command is empty.")

    if shutil.which(command[0]) is None:
        installed_path = Path.home() / ".cargo" / "bin" / command[0]

        if installed_path.is_file():
            command[0] = str(installed_path)

    return command

def build_project(arguments: argparse.Namespace) -> None:
    metadata = tomllib.loads((PROJECT_DIRECTORY / "Cargo.toml").read_text())["package"]
    cargo = resolve_tool("cargo")
    rustc = resolve_tool("rustc")
    mode = arguments.mode
    build_directory = PROJECT_DIRECTORY / "build" / mode
    build_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    revision_process = subprocess.run(
        ["git", "describe", "--always", "--dirty",],
        cwd=PROJECT_DIRECTORY,
        capture_output=True,
        text=True,
        check=False,
    )
    revision = revision_process.stdout.strip() if revision_process.returncode == 0 else "uncommitted"
    environment = os.environ.copy()
    environment["ALEKHINE_REVISION"] = revision
    environment["RUSTC"] = rustc[0]
    flags = ["-Dwarnings",]

    if arguments.is_native:
        flags += ["-C", "target-cpu=native",]

    if environment.get("RUSTFLAGS"):
        flags = shlex.split(environment["RUSTFLAGS"]) + flags

    environment["RUSTFLAGS"] = shlex.join(flags)

    if arguments.has_lto:
        profile = "RELEASE" if mode == "release" else "DEV"
        environment[f"CARGO_PROFILE_{profile}_LTO"] = "true"
        environment[f"CARGO_PROFILE_{profile}_CODEGEN_UNITS"] = "1"

    target_directory = build_directory / "cargo"
    command_options = ["--offline", "--locked", "--target-dir", str(target_directory),]

    if mode == "release":
        command_options += ["--release",]

    run_command(
        cargo + ["build",] + command_options,
        environment,
    )
    executable_suffix = ".exe" if os.name == "nt" else ""
    engine_path = build_directory / ("alekhine" + executable_suffix)
    cargo_profile = "release" if mode == "release" else "debug"
    shutil.copy2(
        target_directory / cargo_profile / engine_path.name,
        engine_path,
    )

    compiler_output = subprocess.check_output(
        rustc + ["--version", "--verbose",],
        text=True,
    ).splitlines()
    compiler_version = compiler_output[0]
    target_architecture = next(line.removeprefix("host: ") for line in compiler_output if line.startswith("host: "))
    source_digest = hashlib.sha256()

    for source_path in sorted([
        *PROJECT_DIRECTORY.glob("src/*.rs"),
        PROJECT_DIRECTORY / "build.rs",
        PROJECT_DIRECTORY / "Cargo.toml",
        PROJECT_DIRECTORY / "Cargo.lock",
    ]):
        source_digest.update(str(source_path.relative_to(PROJECT_DIRECTORY)).encode())
        source_digest.update(source_path.read_bytes())

    with engine_path.open("rb") as engine_file:
        binary_digest = hashlib.file_digest(
            engine_file,
            "sha256",
        ).hexdigest()

    manifest = {
        "name": metadata["name"],
        "version": metadata["version"],
        "revision": revision,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_digest.hexdigest(),
        "binary_sha256": binary_digest,
        "compiler": compiler_version,
        "target_architecture": target_architecture,
        "rust_edition": metadata["edition"],
        "mode": mode,
        "flags": flags,
        "has_link_time_optimization": arguments.has_lto,
        "has_completed_tests": False,
        "has_completed_stockfish_tests": False,
    }
    (build_directory / "build_manifest.json").write_text(json.dumps(
        manifest,
        indent=4,
    ) + "\n")

    if arguments.should_test:
        run_command(
            cargo + ["test",] + command_options,
            environment,
        )
        run_command(
            [sys.executable, str(PROJECT_DIRECTORY / "test" / "uci_tests.py"), str(engine_path),],
            environment,
        )
        run_command(
            [sys.executable, str(PROJECT_DIRECTORY / "test" / "research_tests.py"), str(engine_path),],
            environment,
        )
        run_command(
            [sys.executable, str(PROJECT_DIRECTORY / "test" / "tool_tests.py"),],
            environment,
        )
        integration_command = [sys.executable, str(PROJECT_DIRECTORY / "test" / "integration_tests.py"), str(engine_path),]

        if arguments.stockfish is not None:
            integration_command += ["--stockfish", str(arguments.stockfish),]

        run_command(
            integration_command,
            environment,
        )
        stockfish_training_command = [sys.executable, str(PROJECT_DIRECTORY / "test" / "stockfish_training_tests.py"), str(engine_path),]

        if arguments.stockfish is not None:
            stockfish_training_command += ["--stockfish", str(arguments.stockfish),]

        run_command(
            stockfish_training_command,
            environment,
        )
        run_command(
            [sys.executable, str(PROJECT_DIRECTORY / "test" / "performance_tests.py"), str(engine_path),],
            environment,
        )

    manifest["has_completed_tests"] = arguments.should_test
    manifest["has_completed_stockfish_tests"] = arguments.should_test and (arguments.stockfish is not None)
    (build_directory / "build_manifest.json").write_text(json.dumps(
        manifest,
        indent=4,
    ) + "\n")

    print(f"Built alekhine in {build_directory}.")

def run_build() -> None:
    parser = argparse.ArgumentParser(description="Build and verify the alekhine Rust engine.")
    parser.add_argument(
        "--mode",
        choices=("release", "debug",),
        default="release",
    )
    parser.add_argument(
        "--test",
        dest="should_test",
        action="store_true",
    )
    parser.add_argument(
        "--stockfish",
        type=Path,
        help="Run real Stockfish and python-chess integration workflows with --test.",
    )
    parser.add_argument(
        "--native",
        dest="is_native",
        action="store_true",
    )
    parser.add_argument(
        "--lto",
        dest="has_lto",
        action="store_true",
    )
    arguments = parser.parse_args()

    if (arguments.stockfish is not None) and not arguments.should_test:
        parser.error("The --stockfish option requires --test.")

    try:
        build_project(arguments)
    except (OSError, ValueError, subprocess.CalledProcessError,) as error:
        parser.exit(
            1,
            f"Failed to build alekhine: {error}\n",
        )

if __name__ == "__main__":
    run_build()
