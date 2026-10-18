import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_support import (
    PROJECT_DIRECTORY,
    train_fixture_network,
)

sys.path.insert(
    0,
    str(PROJECT_DIRECTORY / "scripts",),
)

import benchmark
from benchmark_statistics import (
    calculate_distribution,
    compare_measurements,
    summarize_samples,
)

class StatisticsTests(unittest.TestCase,):
    def test_distribution_retains_spread_and_interpolates_percentile(self,) -> None:
        distribution = calculate_distribution([1.0, 2.0, 3.0, 4.0, 5.0,],)

        self.assertEqual(
            distribution["median"],
            3.0,
        )

        self.assertAlmostEqual(
            distribution["p95"],
            4.8,
        )

        self.assertAlmostEqual(
            distribution["standard_deviation"],
            2.5 ** 0.5,
        )

        self.assertEqual(
            distribution["sample_count"],
            5,
        )

    def test_single_sample_and_zero_samples_have_explicit_variance(self,) -> None:
        self.assertEqual(
            calculate_distribution([2.0,],)["standard_deviation"],
            0.0,
        )

        self.assertEqual(
            calculate_distribution([0.0, 0.0,],)["coefficient_of_variation"],
            0.0,
        )

    def test_invalid_samples_are_rejected(self,) -> None:
        for samples in [[], [-1.0,], [float("nan",),], [float("inf",),],]:
            with self.subTest(samples=samples,):
                with self.assertRaises(ValueError,):
                    calculate_distribution(samples,)

    def create_samples(self,) -> list[dict]:
        return [
            {"suite": "search", "name": "fixed_nodes", "model": "bootstrap", "phase": "first pass", "operation_count": 1, "elapsed_nanoseconds": 100000, "node_count": 100,},
            {"suite": "search", "name": "fixed_nodes", "model": "bootstrap", "phase": "measured", "operation_count": 2, "elapsed_nanoseconds": 200, "node_count": 100,},
            {"suite": "search", "name": "fixed_nodes", "model": "bootstrap", "phase": "measured", "operation_count": 2, "elapsed_nanoseconds": 400, "node_count": 100,},
        ]

    def test_summary_excludes_first_pass_and_normalizes_operations(self,) -> None:
        measurements = summarize_samples(self.create_samples(),)
        latency = next((measurement for measurement in measurements if measurement["metric"] == "nanoseconds_per_operation"),)
        throughput = next((measurement for measurement in measurements if measurement["metric"] == "nodes_per_second"),)

        self.assertEqual(
            latency["distribution"]["median"],
            150.0,
        )

        self.assertEqual(
            latency["distribution"]["sample_count"],
            2,
        )

        self.assertEqual(
            throughput["distribution"]["median"],
            375000000.0,
        )

    def test_zero_duration_and_operation_count_are_rejected(self,) -> None:
        for field in ["elapsed_nanoseconds", "operation_count",]:
            samples = self.create_samples()
            samples[1][field] = 0

            with self.assertRaises(ValueError,):
                summarize_samples(samples,)

    def test_regression_gate_distinguishes_improvement_threshold_and_slowdown(self,) -> None:
        baseline = summarize_samples(self.create_samples(),)

        for multiplier, has_regressed in [(0.8, False,), (1.05, False,), (1.5, True,),]:
            samples = self.create_samples()

            for sample in samples:
                sample["elapsed_nanoseconds"] *= multiplier

            comparisons = compare_measurements(
                summarize_samples(samples,),
                baseline,
                10.0,
            )

            self.assertEqual(
                comparisons[0]["has_regressed"],
                has_regressed,
            )

            self.assertAlmostEqual(
                comparisons[0]["change_percent"],
                (multiplier - 1) * 100,
            )

    def test_mismatched_baseline_workloads_are_rejected(self,) -> None:
        baseline = summarize_samples(self.create_samples(),)

        with self.assertRaises(ValueError,):
            compare_measurements(
                [],
                baseline,
                10.0,
            )

    def test_nonfinite_and_negative_thresholds_are_rejected(self,) -> None:
        measurements = summarize_samples(self.create_samples(),)

        for threshold in [-1.0, float("inf",), float("nan",),]:
            with self.assertRaises(ValueError,):
                compare_measurements(
                    measurements,
                    measurements,
                    threshold,
                )

    def create_manifest(self,) -> dict:
        return {"schema_version": 1, "status": "complete", "workloads": {"nodes": 100,}, "suite_sha256": "suite", "network_sha256": None, "environment": {"cpu_model": "CPU",},}

    def test_baseline_compatibility_checks_workload_suite_network_and_status(self,) -> None:
        candidate = self.create_manifest()
        baseline = self.create_manifest()

        benchmark.validate_baseline(
            candidate,
            baseline,
            False,
        )

        for field in ["schema_version", "status", "workloads", "suite_sha256", "network_sha256",]:
            changed = baseline.copy()
            changed[field] = "different"

            with self.assertRaises(ValueError,):
                benchmark.validate_baseline(
                    candidate,
                    changed,
                    False,
                )

    def test_environment_override_is_explicit(self,) -> None:
        candidate = self.create_manifest()
        baseline = self.create_manifest()
        baseline["environment"] = {"cpu_model": "Different CPU",}

        with self.assertRaises(ValueError,):
            benchmark.validate_baseline(
                candidate,
                baseline,
                False,
            )

        benchmark.validate_baseline(
            candidate,
            baseline,
            True,
        )

class HarnessTests(unittest.TestCase,):
    engine_path: Path
    core_binary: Path
    directory: tempfile.TemporaryDirectory

    @classmethod
    def setUpClass(cls,) -> None:
        cls.directory = tempfile.TemporaryDirectory(prefix="alekhine_performance_tests_",)
        cls.core_binary = benchmark.build_core_binary(Path(cls.directory.name,),)

    @classmethod
    def tearDownClass(cls,) -> None:
        cls.directory.cleanup()

    def setUp(self,) -> None:
        self.output_directory = tempfile.TemporaryDirectory(prefix="alekhine_benchmark_artifacts_",)
        self.addCleanup(self.output_directory.cleanup,)
        self.directory_path = Path(self.output_directory.name,)

    def create_arguments(self,) -> argparse.Namespace:
        return argparse.Namespace(
            engine=self.engine_path,
            output=self.directory_path / "results",
            core_binary=self.core_binary,
            network=None,
            suite="all",
            samples=1,
            minimum_ms=1,
            nodes=64,
            depth=1,
            epochs=1,
            batch_size=32,
            training_records=[32,],
            baseline=None,
            maximum_regression_percent=10.0,
            should_allow_environment_change=False,
        )

    def execute_harness(self,) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(self.core_binary,), "--suite", "all", "--samples", "1", "--minimum_ms", "1", "--nodes", "64", "--depth", "1",],
            capture_output=True,
            text=True,
            timeout=60.0,
            check=True,
        )

    def test_real_micro_and_search_samples_validate_budgets_and_stable_checksums(self,) -> None:
        first = [json.loads(line,) for line in self.execute_harness().stdout.splitlines()]
        second = [json.loads(line,) for line in self.execute_harness().stdout.splitlines()]
        search_samples = [sample for sample in first if sample["suite"] == "search"]

        self.assertEqual(
            len(search_samples,),
            432,
        )

        self.assertTrue(all((sample["node_count"] <= 64 for sample in search_samples if "_nodes_" in sample["name"]),),)
        first_outputs = [(sample["name"], sample["best_move"], sample["node_count"], sample["score_centipawns"],) for sample in search_samples]
        second_outputs = [(sample["name"], sample["best_move"], sample["node_count"], sample["score_centipawns"],) for sample in second if sample["suite"] == "search"]

        self.assertEqual(
            first_outputs,
            second_outputs,
        )

        micro_first_passes = {sample["name"]: sample["checksum"] for sample in first if (sample["suite"] == "micro") and (sample["phase"] == "first pass")}

        self.assertEqual(
            micro_first_passes["perft_3_position_0"],
            8902,
        )

        self.assertEqual(
            micro_first_passes["perft_3_position_4"],
            97862,
        )

        self.assertEqual(
            micro_first_passes["perft_3_position_5"],
            2812,
        )

    def test_synthetic_dataset_is_unique_deterministic_and_never_overwritten(self,) -> None:
        paths = [self.directory_path / "first.txt", self.directory_path / "second.txt",]

        for path in paths:
            subprocess.run(
                [str(self.core_binary,), "--dataset", str(path,), "--records", "32",],
                check=True,
                timeout=10.0,
            )

        self.assertEqual(
            paths[0].read_bytes(),
            paths[1].read_bytes(),
        )

        records = paths[0].read_text().splitlines()

        self.assertEqual(
            len(records,),
            32,
        )

        self.assertEqual(
            len({" ".join(record.split()[:4],) for record in records},),
            32,
        )

        self.assertTrue(any((record.endswith("game_3",) for record in records),),)

        failure = subprocess.run(
            [str(self.core_binary,), "--dataset", str(paths[0],), "--records", "32",],
            capture_output=True,
            timeout=10.0,
        )

        self.assertNotEqual(
            failure.returncode,
            0,
        )

        self.assertEqual(
            paths[0].read_bytes(),
            paths[1].read_bytes(),
        )

    def test_full_runner_retains_resources_csv_protocols_and_provenance(self,) -> None:
        arguments = self.create_arguments()

        self.assertIs(
            benchmark.run_benchmarks(arguments,),
            benchmark.BenchmarkOutcome.PASSED,
        )

        report = json.loads((arguments.output / "report.json").read_text(),)

        self.assertEqual(
            report["status"],
            "complete",
        )

        self.assertEqual(
            report["engine_sha256"],
            benchmark.calculate_digest(self.engine_path,),
        )

        self.assertEqual(
            report["core_binary_sha256"],
            benchmark.calculate_digest(self.core_binary,),
        )

        self.assertTrue((arguments.output / "logs" / "latency_bootstrap_1_stop.uci").is_file(),)
        self.assertTrue((arguments.output / "synthetic_32.provenance.json").is_file(),)

        with (arguments.output / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream,),)

        self.assertEqual(
            len(rows,),
            len(report["measurements"],),
        )

        self.assertEqual(
            {row["suite"] for row in rows},
            {"micro", "search", "latency", "training",},
        )

        with self.assertRaises(FileExistsError,):
            benchmark.run_benchmarks(arguments,)

        self.assertEqual(
            json.loads((arguments.output / "report.json").read_text(),)["status"],
            "complete",
        )

    def test_missing_engine_retains_a_failed_manifest(self,) -> None:
        arguments = self.create_arguments()
        arguments.engine = self.directory_path / "missing_engine"

        with self.assertRaises(FileNotFoundError,):
            benchmark.run_benchmarks(arguments,)

        self.assertEqual(
            json.loads((arguments.output / "report.json").read_text(),)["status"],
            "failed",
        )

    def test_trained_nnue_runs_microbenchmarks_and_incremental_invariants(self,) -> None:
        network_path = train_fixture_network(
            self.engine_path,
            self.directory_path,
        )

        completed = subprocess.run(
            [str(self.core_binary,), "--suite", "micro", "--samples", "1", "--minimum_ms", "1", "--network", str(network_path,),],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )

        samples = [json.loads(line,) for line in completed.stdout.splitlines()]
        measurements = summarize_samples(samples,)

        self.assertEqual(
            {measurement["model"] for measurement in measurements},
            {"bootstrap", "nnue",},
        )

        self.assertEqual(
            len(measurements,),
            32,
        )

    def test_real_runner_returns_failed_regression_gate_and_preserves_complete_results(self,) -> None:
        arguments = self.create_arguments()
        arguments.suite = "latency"
        benchmark.run_benchmarks(arguments,)
        baseline_path = arguments.output / "report.json"
        baseline = json.loads(baseline_path.read_text(),)

        for measurement in baseline["measurements"]:
            if measurement["metric"] == "nanoseconds_per_operation":
                measurement["distribution"]["median"] = 0.001

        benchmark.write_json(
            baseline_path,
            baseline,
        )

        arguments.baseline = baseline_path
        arguments.output = self.directory_path / "regression"

        self.assertIs(
            benchmark.run_benchmarks(arguments,),
            benchmark.BenchmarkOutcome.REGRESSED,
        )

        report = json.loads((arguments.output / "report.json").read_text(),)

        self.assertEqual(
            report["status"],
            "complete",
        )

        self.assertTrue(report["has_regressed"],)

    def test_subprocess_failure_retains_logs(self,) -> None:
        with self.assertRaises(subprocess.CalledProcessError,):
            benchmark.execute_command(
                [sys.executable, "-c", "raise SystemExit(7)",],
                self.directory_path,
                "failure",
            )

        self.assertTrue((self.directory_path / "failure.stdout").is_file(),)
        self.assertTrue((self.directory_path / "failure.stderr").is_file(),)

    def test_timeout_terminates_process_group_and_retains_output(self,) -> None:
        with patch.object(
            benchmark,
            "TIMEOUT_SECONDS",
            0.5,
        ):
            with self.assertRaises(subprocess.TimeoutExpired,):
                benchmark.execute_command(
                    [sys.executable, "-c", "import time\nprint(\n    'Started.',\n    flush=True,\n)\ntime.sleep(30)\n",],
                    self.directory_path,
                    "timeout",
                )

        self.assertIn(
            "Started.",
            (self.directory_path / "timeout.stdout").read_text(),
        )

    def test_harness_rejects_unbounded_options(self,) -> None:
        for option, parameter in [("--samples", "0",), ("--nodes", "0",), ("--depth", "7",), ("--records", "100001",), ("--suite", "unknown",),]:
            completed = subprocess.run(
                [str(self.core_binary,), option, parameter,],
                capture_output=True,
                timeout=10.0,
            )

            self.assertNotEqual(
                completed.returncode,
                0,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "engine",
        type=Path,
    )

    arguments, remaining_arguments = parser.parse_known_args()
    HarnessTests.engine_path = arguments.engine.resolve()
    unittest.main(argv=[sys.argv[0],] + remaining_arguments,)
