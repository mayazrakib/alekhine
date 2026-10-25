import argparse
import csv
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import (
    datetime,
    timezone,
)
from enum import Enum
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "scripts"),
)

from benchmark_statistics import (
    Sample,
    compare_measurements,
    summarize_samples,
)
from uci_client import UciClient

from build import (
    PROJECT_DIRECTORY,
    resolve_tool,
)

SCHEMA_VERSION = 1
TIMEOUT_SECONDS = 600.0

class BenchmarkOutcome(Enum,):
    PASSED = "passed"
    REGRESSED = "regressed"

def calculate_digest(path: Path,) -> str:
    digest = hashlib.sha256()

    with path.open("rb",) as stream:
        while chunk := stream.read(1_048_576,):
            digest.update(chunk,)

    return digest.hexdigest()

def write_json(
    path: Path,
    document: object,
) -> None:
    path.write_text(json.dumps(
        document,
        indent=4,
        allow_nan=False,
    ) + "\n",)

def build_core_binary(log_directory: Path,) -> Path:
    environment = os.environ.copy()
    environment["RUSTC"] = resolve_tool("rustc",)[0]

    environment["RUSTFLAGS"] = environment.get(
        "RUSTFLAGS",
        "",
    ) + " -Dwarnings"

    command = resolve_tool("cargo",) + [
        "build",
        "--offline",
        "--locked",
        "--release",
        "--bench",
        "performance",
        "--message-format=json",
        "--target-dir",
        str(PROJECT_DIRECTORY / "build" / "benchmark" / "cargo",),
    ]

    completed = subprocess.run(
        command,
        cwd=PROJECT_DIRECTORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )

    (log_directory / "core_build.jsonl").write_text(completed.stdout,)
    (log_directory / "core_build.stderr").write_text(completed.stderr,)
    completed.check_returncode()

    for line in completed.stdout.splitlines():
        event = json.loads(line,)

        if (event.get("reason",) == "compiler-artifact") and (event["target"]["name"] == "performance") and event.get("executable",):
            return Path(event["executable"],)

    raise RuntimeError("Failed to locate the compiled performance harness.",)

def read_environment() -> dict:
    cpu_model = platform.processor()
    cpu_path = Path("/proc/cpuinfo",)

    if cpu_path.is_file():
        for line in cpu_path.read_text().splitlines():
            if line.startswith("model name",):
                cpu_model = line.split(
                    ":",
                    1,
                )[1].strip()

                break

    affinity = sorted(os.sched_getaffinity(0,),) if hasattr(
        os,
        "sched_getaffinity",
    ) else None

    return {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "python": platform.python_version(),
        "performance_clock": vars(time.get_clock_info("perf_counter",),),
    }

def read_process_resources(process_id: int,) -> tuple[
    float | None,
    int | None,
]:
    directory = Path("/proc",) / str(process_id,)

    if not directory.is_dir():
        return None, None

    fields = (directory / "stat").read_text().rsplit(
        ")",
        1,
    )[1].split()

    cpu_seconds = (int(fields[11],) + int(fields[12],)) / os.sysconf("SC_CLK_TCK",)
    peak_rss_bytes = None

    for line in (directory / "status").read_text().splitlines():
        if line.startswith("VmHWM:",):
            peak_rss_bytes = int(line.split()[1],) * 1024

    return cpu_seconds, peak_rss_bytes

def execute_command(
    command: list[str],
    log_directory: Path,
    name: str,
) -> tuple[
    str,
    int,
    dict,
]:
    resource_path = log_directory / f"{name}.resources.json"
    time_path = Path("/usr/bin/time",)

    if time_path.is_file() and (platform.system() == "Linux"):
        resource_format = '{"user_seconds":%U,"system_seconds":%S,"peak_rss_kib":%M,"major_fault_count":%F,"minor_fault_count":%R,"voluntary_switch_count":%w,"involuntary_switch_count":%c,"exit_code":%x}'
        command = [str(time_path,), "-q", "-o", str(resource_path,), "-f", resource_format, "--",] + command

    started_at = time.perf_counter_ns()

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIRECTORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=(os.name == "posix"),
    )

    failure = None

    try:
        stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS,)
    except subprocess.TimeoutExpired as error:
        failure = error

        if os.name == "posix":
            os.killpg(
                process.pid,
                signal.SIGKILL,
            )
        else:
            process.kill()

        stdout, stderr = process.communicate()

    elapsed_nanoseconds = time.perf_counter_ns() - started_at
    (log_directory / f"{name}.stdout").write_text(stdout,)
    (log_directory / f"{name}.stderr").write_text(stderr,)

    if failure:
        raise failure

    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )

    resources = json.loads(resource_path.read_text(),) if resource_path.is_file() else {}

    return stdout, elapsed_nanoseconds, resources

def append_sample(
    output: Path,
    samples: list[Sample],
    sample: Sample,
) -> None:
    with (output / "samples.jsonl").open("a",) as stream:
        stream.write(json.dumps(
            sample,
            allow_nan=False,
        ) + "\n",)

    samples.append(sample,)

def run_core_suite(
    arguments: argparse.Namespace,
    core_binary: Path,
    samples: list[Sample],
) -> None:
    command = [
        str(core_binary,),
        "--samples",
        str(arguments.samples,),
        "--minimum_ms",
        str(arguments.minimum_ms,),
        "--nodes",
        str(arguments.nodes,),
        "--depth",
        str(arguments.depth,),
        "--suite",
        arguments.suite if arguments.suite in ["micro", "search",] else "all",
    ]

    if arguments.network:
        command += ["--network", str(arguments.network,),]

    stdout, elapsed_nanoseconds, resources = execute_command(
        command,
        arguments.output / "logs",
        "core",
    )

    write_json(
        arguments.output / "core_process.json",
        {"elapsed_nanoseconds": elapsed_nanoseconds, "resources": resources,},
    )

    for line in stdout.splitlines():
        sample = json.loads(line,)

        if sample["suite"] == "metadata":
            append_sample(
                arguments.output,
                samples,
                sample,
            )

            continue

        if (sample["elapsed_nanoseconds"] <= 0) or (sample["operation_count"] <= 0) or (sample["phase"] not in ["measured", "first pass",]):
            raise ValueError("Failed to validate a performance harness sample.",)

        append_sample(
            arguments.output,
            samples,
            sample,
        )

def run_latency_suite(
    arguments: argparse.Namespace,
    samples: list[Sample],
) -> None:
    legal_output, _, _ = execute_command(
        [str(arguments.engine,), "divide", "1",],
        arguments.output / "logs",
        "latency_legal_moves",
    )

    legal_moves = {line.split()[0] for line in legal_output.splitlines() if (len(line.split(),) == 2) and line.split()[1].isdigit()}

    if len(legal_moves,) != 20:
        raise RuntimeError("Failed to validate the latency workload's initial legal moves.",)

    models = [("bootstrap", None,),]

    if arguments.network:
        models.append(("nnue", arguments.network,),)

    for model, network_path in models:
        options = {"Hash": "16", "Move Overhead": "0",}

        if network_path:
            options["EvalFile"] = str(network_path,)

        for sample_index in range(arguments.samples + 1,):
            phase = "first pass" if sample_index == 0 else "measured"
            started_at = time.perf_counter_ns()

            client = UciClient(
                arguments.engine,
                options,
            )

            startup_nanoseconds = time.perf_counter_ns() - started_at

            try:
                append_sample(
                    arguments.output,
                    samples,
                    {"suite": "latency", "name": "startup_and_model_ready", "model": model, "phase": phase, "sample_index": sample_index, "operation_count": 1, "elapsed_nanoseconds": startup_nanoseconds,},
                )

                for name in ["idle_ready", "active_ready", "stop", "ponderhit", "movetime_50",]:
                    client.start_game()
                    client.send_command("position startpos",)
                    transcript = []

                    if name in ["active_ready", "stop", "ponderhit",]:
                        client.is_searching = True
                        client.send_command("go ponder nodes 1000" if name == "ponderhit" else "go infinite",)

                        transcript += client.wait_for_line(
                            "info depth ",
                            20.0,
                        )

                    cpu_before, _ = read_process_resources(client.process.pid,)
                    started_at = time.perf_counter_ns()
                    command = {"idle_ready": "isready", "active_ready": "isready", "stop": "stop", "ponderhit": "ponderhit", "movetime_50": "go movetime 50",}[name]
                    client.send_command(command,)

                    transcript += client.wait_for_line(
                        "readyok" if name.endswith("ready",) else "bestmove ",
                        20.0,
                    )

                    elapsed_nanoseconds = time.perf_counter_ns() - started_at
                    cpu_after, peak_rss_bytes = read_process_resources(client.process.pid,)

                    if not name.endswith("ready",):
                        client.is_searching = False
                        best_move = transcript[-1].split()[1]

                        if best_move not in legal_moves:
                            raise RuntimeError("Failed to validate a latency benchmark best move.",)

                    sample: Sample = {
                        "suite": "latency",
                        "name": name,
                        "model": model,
                        "phase": phase,
                        "sample_index": sample_index,
                        "operation_count": 1,
                        "elapsed_nanoseconds": elapsed_nanoseconds,
                        "cpu_seconds": cpu_after - cpu_before if (cpu_after is not None) and (cpu_before is not None) else None,
                        "peak_rss_bytes": peak_rss_bytes,
                    }

                    if name == "movetime_50":
                        sample["overshoot_nanoseconds"] = max(
                            0,
                            elapsed_nanoseconds - 50_000_000,
                        )

                    append_sample(
                        arguments.output,
                        samples,
                        sample,
                    )

                    (arguments.output / "logs" / f"latency_{model}_{sample_index}_{name}.uci").write_text("\n".join(transcript,) + "\n",)
            finally:
                client.close_process()

def run_training_suite(
    arguments: argparse.Namespace,
    core_binary: Path,
    samples: list[Sample],
) -> None:
    for record_count in arguments.training_records:
        dataset_path = arguments.output / f"synthetic_{record_count}.txt"

        execute_command(
            [str(core_binary,), "--dataset", str(dataset_path,), "--records", str(record_count,),],
            arguments.output / "logs",
            f"generate_{record_count}",
        )

        dataset_digest = calculate_digest(dataset_path,)

        write_json(
            dataset_path.with_suffix(".provenance.json",),
            {"is_synthetic": True, "seed": 20261004, "record_count": record_count, "sha256": dataset_digest, "targets": "Material score, synthetic draw outcome, and first legal move.",},
        )

        for sample_index in range(arguments.samples + 1,):
            network_path = arguments.output / f"trained_{record_count}_{sample_index}.nnue"

            stdout, elapsed_nanoseconds, resources = execute_command(
                [str(arguments.engine,), "train_network", str(dataset_path,), str(network_path,), str(arguments.epochs,), "--batch_size", str(arguments.batch_size,), "--seed", "20261004", "--maximum_record_count", str(record_count,),],
                arguments.output / "logs",
                f"train_{record_count}_{sample_index}",
            )

            training_manifest = json.loads(network_path.with_suffix(".training.json",).read_text(),)

            if not network_path.is_file() or not stdout or (training_manifest["training_record_count"] + training_manifest["validation_record_count"] != record_count):
                raise RuntimeError("Failed to validate a training benchmark export.",)

            append_sample(
                arguments.output,
                samples,
                {
                    "suite": "training",
                    "name": f"records_{record_count}_batch_{arguments.batch_size}",
                    "model": "nnue",
                    "phase": "first pass" if sample_index == 0 else "measured",
                    "sample_index": sample_index,
                    "operation_count": 1,
                    "elapsed_nanoseconds": elapsed_nanoseconds,
                    "record_count": record_count,
                    "epoch_count": arguments.epochs,
                    "cpu_seconds": resources["user_seconds"] + resources["system_seconds"] if resources else None,
                    "peak_rss_bytes": resources["peak_rss_kib"] * 1024 if resources else None,
                },
            )

def validate_baseline(
    candidate: dict,
    baseline: dict,
    should_allow_environment_change: bool,
) -> None:
    if not isinstance(
        baseline,
        dict,
    ):
        raise ValueError("Failed to compare a baseline that is not a report object.",)

    if (baseline.get("schema_version",) != SCHEMA_VERSION) or (baseline.get("status",) != "complete"):
        raise ValueError("Failed to compare an incomplete or incompatible baseline report.",)

    for field in ["workloads", "suite_sha256", "network_sha256",]:
        if candidate[field] != baseline[field]:
            raise ValueError(f"Failed to compare benchmarks with mismatched {field}.",)

    if not should_allow_environment_change and (candidate["environment"] != baseline["environment"]):
        raise ValueError("Failed to compare benchmarks from different environments. Use --allow_environment_change for an explicit exploratory comparison.",)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure engine performance and retain reproducible benchmark artifacts.",)

    parser.add_argument(
        "--engine",
        type=Path,
        default=PROJECT_DIRECTORY / "build" / "release" / "alekhine",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--core_binary",
        type=Path,
    )

    parser.add_argument(
        "--network",
        type=Path,
    )

    parser.add_argument(
        "--suite",
        choices=["all", "micro", "search", "latency", "training",],
        default="all",
    )

    for name, default in [("samples", 5,), ("minimum_ms", 15,), ("nodes", 2000,), ("depth", 3,), ("epochs", 2,), ("batch_size", 256,),]:
        parser.add_argument(
            f"--{name}",
            type=int,
            default=default,
        )

    parser.add_argument(
        "--training_records",
        type=int,
        nargs="+",
        default=[32, 256, 2048,],
    )

    parser.add_argument(
        "--baseline",
        type=Path,
    )

    parser.add_argument(
        "--maximum_regression_percent",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--allow_environment_change",
        dest="should_allow_environment_change",
        action="store_true",
    )

    arguments = parser.parse_args()

    for name, minimum, maximum in [("samples", 1, 100,), ("minimum_ms", 1, 1000,), ("nodes", 32, 1_000_000,), ("depth", 1, 6,), ("epochs", 1, 100,), ("batch_size", 1, 8192,),]:
        if not minimum <= getattr(
            arguments,
            name,
        ) <= maximum:
            parser.error(f"Failed to validate {name}: expected {minimum} through {maximum}.",)

    if not arguments.training_records or len(arguments.training_records,) > 8 or len(set(arguments.training_records,),) != len(arguments.training_records,) or any((not 32 <= count <= 100_000 for count in arguments.training_records),):
        parser.error("Failed to validate unique training record counts between 32 and 100000.",)

    if not math.isfinite(arguments.maximum_regression_percent,) or (arguments.maximum_regression_percent < 0):
        parser.error("Failed to validate the regression threshold.",)

    arguments.engine = arguments.engine.resolve()
    arguments.output = arguments.output.resolve()

    for name in ["network", "core_binary", "baseline",]:
        if getattr(
            arguments,
            name,
        ):
            setattr(
                arguments,
                name,
                getattr(
                    arguments,
                    name,
                ).resolve(),
            )

    return arguments

def run_benchmarks(arguments: argparse.Namespace,) -> BenchmarkOutcome:
    arguments.output.mkdir(parents=True,)
    log_directory = arguments.output / "logs"
    log_directory.mkdir()

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now(timezone.utc,).isoformat(),
        "environment": read_environment(),
        "workloads": {name: getattr(
            arguments,
            name,
        ) for name in ["suite", "samples", "minimum_ms", "nodes", "depth", "epochs", "batch_size", "training_records",]},
        "suite_sha256": hashlib.sha256(b"".join(((PROJECT_DIRECTORY / path).read_bytes() for path in ["tests/perf/performance.rs", "test/openings.epd", "tests/perf/benchmark.py", "tests/perf/benchmark_statistics.py",]),),).hexdigest(),
        "resource_scope": "GNU Time measures whole subprocesses. Linux proc CPU deltas use clock ticks, and RSS is the process high-water mark. Unsupported resource fields are null.",
        "training_scope": "Synthetic workload throughput includes loading, feature encoding, optimizer epochs, validation, quantization, and export. It is not model-quality evidence.",
    }

    manifest_path = arguments.output / "report.json"

    write_json(
        manifest_path,
        report,
    )

    try:
        report["engine_sha256"] = calculate_digest(arguments.engine,)
        report["network_sha256"] = calculate_digest(arguments.network,) if arguments.network else None
        report["core_scope"] = "The harness measures its linked Rust library. Supply the archived --core_binary when comparing an older engine implementation. The UCI and training suites use --engine."
        baseline = json.loads(arguments.baseline.read_text(),) if arguments.baseline else None

        if baseline is not None:
            validate_baseline(
                report,
                baseline,
                arguments.should_allow_environment_change,
            )

        core_binary = None

        if arguments.suite != "latency":
            core_binary = arguments.core_binary or build_core_binary(log_directory,)
            report["core_binary_sha256"] = calculate_digest(core_binary,)
            report["core_binary"] = str(core_binary,)

        report["engine"] = str(arguments.engine,)

        report["compiler"] = subprocess.check_output(
            resolve_tool("rustc",) + ["--version",],
            text=True,
            timeout=20.0,
        ).strip()

        report["rustflags"] = os.environ.get(
            "RUSTFLAGS",
            "",
        )

        report["core_effective_rustflags"] = report["rustflags"] + " -Dwarnings" if not arguments.core_binary else None
        report["cargo_profile_overrides"] = {name: setting for name, setting in os.environ.items() if name.startswith("CARGO_PROFILE_",)}
        report["core_source_sha256"] = hashlib.sha256(b"".join((path.read_bytes() for path in sorted((PROJECT_DIRECTORY / "src").glob("*.rs",),)),),).hexdigest() if not arguments.core_binary else None
        report["network"] = str(arguments.network,) if arguments.network else None
        build_manifest_path = arguments.engine.parent / "build_manifest.json"
        report["engine_build_manifest"] = json.loads(build_manifest_path.read_text(),) if build_manifest_path.is_file() else None

        if report["engine_build_manifest"] and (report["engine_build_manifest"]["binary_sha256"] != report["engine_sha256"]):
            raise ValueError("Failed to benchmark an engine with a stale build manifest.",)

        samples: list[Sample] = []

        if arguments.suite in ["all", "micro", "search",]:
            print(
                "Measuring engine operations and search.",
                flush=True,
            )

            run_core_suite(
                arguments,
                core_binary,
                samples,
            )

        if arguments.suite in ["all", "latency",]:
            print(
                "Measuring UCI lifecycle latency.",
                flush=True,
            )

            run_latency_suite(
                arguments,
                samples,
            )

        if arguments.suite in ["all", "training",]:
            print(
                "Measuring synthetic training workloads.",
                flush=True,
            )

            run_training_suite(
                arguments,
                core_binary,
                samples,
            )

        if (calculate_digest(arguments.engine,) != report["engine_sha256"]) or (core_binary and (calculate_digest(core_binary,) != report["core_binary_sha256"])) or (arguments.network and (calculate_digest(arguments.network,) != report["network_sha256"])):
            raise RuntimeError("Failed to complete benchmarks because a measured binary or network changed.",)

        measurements = summarize_samples(samples,)

        if not measurements:
            raise RuntimeError("Failed to complete benchmarks without measured samples.",)

        report["measurements"] = measurements

        report["comparisons"] = compare_measurements(
            measurements,
            baseline["measurements"],
            arguments.maximum_regression_percent,
        ) if baseline is not None else []

        report["maximum_regression_percent"] = arguments.maximum_regression_percent
        report["has_regressed"] = any((comparison["has_regressed"] for comparison in report["comparisons"]),)
        report["status"] = "complete"
        report["completed_at"] = datetime.now(timezone.utc,).isoformat()

        with (arguments.output / "summary.csv").open(
            "w",
            newline="",
        ) as stream:
            writer = csv.writer(stream,)
            writer.writerow(["suite", "name", "model", "metric", "sample_count", "median", "p95", "coefficient_of_variation",],)

            for measurement in measurements:
                distribution = measurement["distribution"]
                writer.writerow([measurement[field] for field in ["suite", "name", "model", "metric",]] + [distribution[field] for field in ["sample_count", "median", "p95", "coefficient_of_variation",]],)

        write_json(
            manifest_path,
            report,
        )

        print(f"Completed benchmarks. Results are in {arguments.output}.",)

        return BenchmarkOutcome.REGRESSED if report["has_regressed"] else BenchmarkOutcome.PASSED
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = str(error,)

        write_json(
            manifest_path,
            report,
        )

        raise

def main() -> int:
    try:
        outcome = run_benchmarks(parse_arguments(),)

        if outcome is BenchmarkOutcome.REGRESSED:
            print(
                "Detected benchmark regressions exceeding the configured threshold.",
                file=sys.stderr,
            )

            return 2

        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,) as error:
        print(
            f"Failed to complete performance testing: {error}",
            file=sys.stderr,
        )

        return 1

if __name__ == "__main__":
    sys.exit(main(),)
