import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from test_support import (
    PROJECT_DIRECTORY,
    create_uci_fixture,
)

sys.path.insert(
    0,
    str(PROJECT_DIRECTORY / "tools" / "scripts",),
)

import match
import package
from uci_client import (
    EngineOptions,
    UciClient,
    parse_variations,
)

class ProtocolClientTests(unittest.TestCase,):
    def setUp(self,) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="alekhine_protocol_tests_",)
        self.addCleanup(self.directory.cleanup,)
        self.fixture_path = create_uci_fixture(Path(self.directory.name,),)

    def create_client(
        self,
        options: EngineOptions,
    ) -> UciClient:
        client = UciClient(
            self.fixture_path,
            options,
        )

        self.addCleanup(client.close_process,)

        return client

    def test_handshake_search_and_cleanup_reap_process_and_threads(self,) -> None:
        client = self.create_client({"Hash": "1",},)

        best_move, transcript, elapsed_seconds = client.search_position(
            "position startpos",
            "go depth 2",
            5.0,
        )

        self.assertEqual(
            best_move,
            "e2e4",
        )

        self.assertEqual(
            len(parse_variations(transcript,),),
            1,
        )

        self.assertGreaterEqual(
            elapsed_seconds,
            0.0,
        )

        client.close_process()
        self.assertIsNotNone(client.process.poll(),)
        self.assertFalse(client.output_thread.is_alive(),)
        self.assertFalse(client.error_thread.is_alive(),)

    def test_timeout_can_be_stopped_and_followed_by_a_new_game(self,) -> None:
        client = self.create_client({"fault_mode": "silent",},)

        with self.assertRaises(TimeoutError,):
            client.search_position(
                "position startpos",
                "go infinite",
                0.05,
            )

        self.assertTrue(client.is_searching,)
        client.start_game()
        self.assertFalse(client.is_searching,)
        client.send_command("setoption name fault_mode value normal",)

        best_move, transcript, elapsed_seconds = client.search_position(
            "position startpos",
            "go depth 1",
            5.0,
        )

        self.assertEqual(
            best_move,
            "e2e4",
        )

    def test_crash_reports_original_diagnostics(self,) -> None:
        client = self.create_client({"fault_mode": "crash",},)

        with self.assertRaises(RuntimeError,):
            client.search_position(
                "position startpos",
                "go depth 1",
                5.0,
            )

        client.close_process()

        self.assertEqual(
            client.process.returncode,
            1,
        )

        self.assertIn(
            "Forced fixture termination.",
            client.diagnostics,
        )

    def test_oversized_output_fails_with_a_bounded_diagnostic(self,) -> None:
        client = self.create_client({"fault_mode": "oversized",},)

        with self.assertRaises(RuntimeError,):
            client.search_position(
                "position startpos",
                "go depth 1",
                5.0,
            )

        self.assertIn(
            "Rejected oversized engine output.",
            client.diagnostics,
        )

    def test_option_rejection_is_reported_during_readiness(self,) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Failed to configure",
        ):
            UciClient(
                self.fixture_path,
                {"fault_mode": "invalid option",},
            )

    def test_commands_reject_newline_injection_and_byte_limit_overflow(self,) -> None:
        client = self.create_client({},)

        for command in ["isready\nquit", "isready\rquit", "x" * 65537, "é" * 32769,]:
            with self.subTest(command_length=len(command,),):
                with self.assertRaises(ValueError,):
                    client.send_command(command,)

        client.send_command("isready",)

        self.assertEqual(
            client.wait_for_line(
                "readyok",
                5.0,
            )[-1],
            "readyok",
        )

    def test_variation_parser_rejects_malformed_reports_and_bound_scores(self,) -> None:
        self.assertEqual(
            parse_variations([
                "info depth broken score cp 1 pv e2e4",
                "info depth 1 score cp broken pv e2e4",
                "info depth 1 score cp 1 lowerbound pv e2e4",
                "info depth 1 score cp 1 upperbound pv e2e4",
                "info depth 1 score cp 1 pv",
                "info string depth 1 score cp 1 pv e2e4",
                "info depth -1 score cp 1 pv e2e4",
                "info depth 1 multipv 0 score cp 1 pv e2e4",
            ],),
            [],
        )

    def test_variation_parser_accepts_mate_scores_and_sorts_multipv(self,) -> None:
        reports = parse_variations([
            "info depth 2 multipv 2 score cp -50 pv d2d4",
            "info depth 2 multipv 1 score mate -3 pv e2e4 e7e5",
        ],)

        self.assertEqual(
            [report["variation_index"] for report in reports],
            [1, 2,],
        )

        self.assertEqual(
            (reports[0]["score_kind"], reports[0]["score"],),
            ("mate", -3,),
        )

class MatchCommandTests(unittest.TestCase,):
    def setUp(self,) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="alekhine_match_command_",)
        self.addCleanup(self.directory.cleanup,)
        self.root = Path(self.directory.name,)
        self.candidate = create_uci_fixture(self.root,)
        self.baseline = self.root / "baseline_fixture"
        self.baseline.write_bytes(self.candidate.read_bytes(),)
        self.baseline.chmod(0o700,)
        self.openings = self.root / "openings.epd"
        self.openings.write_bytes((PROJECT_DIRECTORY / "tests" / "openings.epd").read_bytes(),)

    def create_arguments(self,) -> argparse.Namespace:
        return argparse.Namespace(
            candidate=self.candidate,
            baseline=self.baseline,
            candidate_option=[],
            baseline_option=[],
            openings=self.openings,
            output=self.root / "results",
            runner="fastchess",
            time_control="10+0.1",
            rounds=10,
            concurrency=1,
            hash_mib=16,
            seed=42,
            elo1=5.0,
            alpha=0.05,
            beta=0.05,
        )

    def test_command_contains_paired_openings_pentanomial_and_sprt_settings(self,) -> None:
        command = match.build_match_command(self.create_arguments(),)

        for token in ["-repeat", "penta=true", "elo0=0", "elo1=5.0", "order=random", "format=epd",]:
            self.assertIn(
                token,
                command,
            )

        self.assertEqual(
            command.count("option.Hash=16",),
            2,
        )

        self.assertFalse((self.root / "results").exists(),)

    def test_identical_binary_requires_distinct_option_configurations(self,) -> None:
        arguments = self.create_arguments()
        arguments.baseline = arguments.candidate

        with self.assertRaises(ValueError,):
            match.build_match_command(arguments,)

        arguments.candidate_option = ["Policy Simulations=0",]

        self.assertIn(
            "option.Policy Simulations=0",
            match.build_match_command(arguments,),
        )

    def test_match_preparation_rejects_invalid_budgets_and_sprt_parameters(self,) -> None:
        for name, invalid_setting in [("rounds", 0,), ("rounds", 1000001,), ("hash_mib", 0,), ("hash_mib", 513,), ("concurrency", 0,), ("concurrency", 33,), ("elo1", 0.0,), ("elo1", float("nan",),), ("alpha", 0.0,), ("beta", 0.5,),]:
            with self.subTest(
                parameter=name,
                setting=invalid_setting,
            ):
                arguments = self.create_arguments()

                setattr(
                    arguments,
                    name,
                    invalid_setting,
                )

                with self.assertRaises(ValueError,):
                    match.build_match_command(arguments,)

    def test_match_preparation_preserves_existing_output(self,) -> None:
        arguments = self.create_arguments()
        arguments.output.mkdir()
        marker = arguments.output / "retained.txt"
        marker.write_text("Preserved results.\n",)

        with self.assertRaises(ValueError,):
            match.build_match_command(arguments,)

        self.assertEqual(
            marker.read_text(),
            "Preserved results.\n",
        )

class PackageTests(unittest.TestCase,):
    def setUp(self,) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="alekhine_package_tests_",)
        self.addCleanup(self.directory.cleanup,)
        self.root = Path(self.directory.name,)
        build_directory = self.root / "build/release"
        build_directory.mkdir(parents=True,)
        self.binary_path = build_directory / "alekhine"
        self.binary_path.write_bytes(b"Test executable payload.\n",)

        manifest = {
            "version": "0.1.0",
            "target_architecture": "test_target",
            "binary_sha256": hashlib.sha256(self.binary_path.read_bytes(),).hexdigest(),
        }

        (build_directory / "build_manifest.json").write_text(json.dumps(manifest,),)

        for filename in ["LICENSE", "README.md", "project.toml", "Cargo.toml",]:
            (self.root / filename).write_text(f"Test {filename} content.\n",)

        self.arguments = argparse.Namespace(
            mode="release",
            output=self.root / "dist",
        )

        self.project_patch = patch.object(
            package,
            "PROJECT_DIRECTORY",
            self.root,
        )

        self.project_patch.start()
        self.addCleanup(self.project_patch.stop,)

    def test_archives_preserve_binary_and_manifest_bytes(self,) -> None:
        package.package_project(self.arguments,)
        archive_prefix = "alekhine-0.1.0-test_target-release"

        with tarfile.open(self.arguments.output / f"{archive_prefix}.tar.gz",) as archive:
            self.assertEqual(
                archive.extractfile(f"{archive_prefix}/bin/alekhine",).read(),
                self.binary_path.read_bytes(),
            )

            self.assertIn(
                f"{archive_prefix}/build_manifest.json",
                archive.getnames(),
            )

        with zipfile.ZipFile(self.arguments.output / f"{archive_prefix}.zip",) as archive:
            self.assertEqual(
                archive.read(f"{archive_prefix}/bin/alekhine",),
                self.binary_path.read_bytes(),
            )

    def test_packaging_rejects_digest_mismatch_before_creating_archives(self,) -> None:
        self.binary_path.write_bytes(b"Changed executable payload.\n",)

        with self.assertRaises(ValueError,):
            package.package_project(self.arguments,)

        self.assertFalse(self.arguments.output.exists(),)

    def test_packaging_refuses_to_overwrite_existing_archives(self,) -> None:
        package.package_project(self.arguments,)
        archive_path = next(self.arguments.output.glob("*.zip",),)
        original_bytes = archive_path.read_bytes()

        with self.assertRaises(ValueError,):
            package.package_project(self.arguments,)

        self.assertEqual(
            archive_path.read_bytes(),
            original_bytes,
        )

if __name__ == "__main__":
    unittest.main()
