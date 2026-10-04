import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict

from test_support import (
    PROJECT_DIRECTORY,
    TRAINING_RECORDS,
    create_uci_fixture,
    train_fixture_network,
)

sys.path.insert(
    0,
    str(PROJECT_DIRECTORY / "scripts"),
)

from uci_client import (
    EngineOptions,
    UciClient,
    parse_variations,
)

START_FEN = TRAINING_RECORDS.split("|")[0].strip()
TEACHER_PGN = (
    '[Event "Teacher fixture one"]\n[Result "*"]\n\n'
    '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O *\n\n'
    '[Event "Teacher fixture two"]\n[Result "*"]\n\n'
    '1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 7. Bh4 b6 8. Bd3 Bb7 *\n'
)

class MatchGame(TypedDict):
    pair_index: int
    candidate_color: str
    result: str
    candidate_score: float
    termination: str
    ply_count: int
    has_forfeit: bool

class MatchSummary(TypedDict):
    wins: int
    draws: int
    losses: int
    score_fraction: float
    games: list[MatchGame]
    pentanomial_counts: list[int]
    is_complete: bool

class ProcessTests(unittest.TestCase):
    engine_path = PROJECT_DIRECTORY / "build/release/alekhine"
    stockfish_path = Path("/usr/bin/stockfish")

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="alekhine_integration_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_program(
        self,
        arguments: list[str],
        is_success_expected: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=60.0,
        )

        if is_success_expected:
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
        else:
            self.assertNotEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )

        return completed

    def create_client(
        self,
        options: EngineOptions,
    ) -> UciClient:
        client = UciClient(
            self.engine_path,
            options,
        )
        self.addCleanup(client.close_process)

        return client

class EngineIntegrationTests(ProcessTests):
    def test_cli_identity_and_unknown_command_exit_status(self) -> None:
        identity = self.run_program([str(self.engine_path), "version",])
        self.assertIn(
            "alekhine 0.1.0",
            identity.stdout,
        )
        self.assertEqual(
            identity.stderr,
            "",
        )
        rejected = self.run_program(
            [str(self.engine_path), "unknown_command",],
            False,
        )
        self.assertEqual(
            rejected.returncode,
            2,
        )
        self.assertIn(
            "Use alekhine",
            rejected.stderr,
        )

    def test_perft_and_divide_agree_on_reference_leaf_counts(self) -> None:
        perft = self.run_program([str(self.engine_path), "perft", "2",])
        divide = self.run_program([str(self.engine_path), "divide", "2",])
        self.assertIn(
            "Counted 400 leaf nodes",
            perft.stdout,
        )
        counts = {}

        for line in divide.stdout.splitlines():
            if re.fullmatch(
                r"[a-h][1-8][a-h][1-8][nbrq]? \d+",
                line,
            ):
                move_name, count_text = line.split()
                counts[move_name] = int(count_text)

        self.assertEqual(
            len(counts),
            20,
        )
        self.assertEqual(
            sum(counts.values()),
            400,
        )
        self.assertTrue(all(count == 20 for count in counts.values()))

    def test_cli_rejects_invalid_depths_and_fen(self) -> None:
        for arguments in [["perft", "-1",], ["perft", "127",], ["perft", "NaN",], ["perft", "1", "Invalid FEN.",],]:
            completed = self.run_program(
                [str(self.engine_path), *arguments,],
                False,
            )
            self.assertIn(
                "Failed",
                completed.stderr,
            )

    def test_benchmark_has_reproducible_nodes_and_checksum(self) -> None:
        measurements = []

        for _ in range(2):
            completed = self.run_program([str(self.engine_path), "bench", "1",])
            measurements.append([line for line in completed.stdout.splitlines() if line.startswith(("Nodes:", "Checksum:", "Network fingerprint:",))])

        self.assertEqual(
            measurements[0],
            measurements[1],
        )
        self.assertEqual(
            len(measurements[0]),
            3,
        )

    def test_bootstrap_export_is_deterministic_and_loads_paths_with_spaces(self) -> None:
        network_path = self.root / "bootstrap weights.network"
        self.run_program([str(self.engine_path), "export_network", str(network_path),])
        original_bytes = network_path.read_bytes()
        self.run_program([str(self.engine_path), "export_network", str(network_path),])
        self.assertEqual(
            network_path.read_bytes(),
            original_bytes,
        )
        client = self.create_client({"EvalFile": str(network_path),})
        best_move, transcript, elapsed_seconds = client.search_position(
            "position startpos",
            "go searchmoves e2e4 depth 1",
            10.0,
        )
        self.assertEqual(
            best_move,
            "e2e4",
        )

    def test_training_cli_rejects_duplicate_unknown_and_missing_options(self) -> None:
        dataset_path = self.root / "positions.txt"
        network_path = self.root / "model.nnue"
        dataset_path.write_text(TRAINING_RECORDS)

        for options in [["--batch_size",], ["--unknown", "1",], ["--seed", "1", "--seed", "2",], ["--batch_size", "0",], ["--learning_rate", "NaN",], ["--learning_rate", "inf",], ["--maximum_record_count", "2",],]:
            self.run_program(
                [str(self.engine_path), "train_network", str(dataset_path), str(network_path), "1", *options,],
                False,
            )
            self.assertFalse(network_path.exists())
            self.assertFalse(network_path.with_suffix(".training.json").exists())

    def test_training_manifest_matches_loaded_network_and_dataset(self) -> None:
        network_path = train_fixture_network(
            self.engine_path,
            self.root,
        )
        manifest = json.loads(network_path.with_suffix(".training.json").read_text())
        self.assertEqual(
            (manifest["training_record_count"], manifest["validation_record_count"],),
            (2, 1,),
        )
        self.assertTrue(manifest["has_trained_policy"])
        self.assertTrue(manifest["has_trained_outcome"])
        client = self.create_client({"EvalFile": str(network_path), "Policy Simulations": "0",})
        client.send_command(f"setoption name EvalFile value {network_path}")
        client.send_command("isready")
        transcript = client.wait_for_line(
            "readyok",
            10.0,
        )
        self.assertTrue(any(str(manifest["network_fingerprint"]) in line for line in transcript))

    def test_trained_multipv_reports_distinct_legal_roots_within_node_budget(self) -> None:
        network_path = train_fixture_network(
            self.engine_path,
            self.root,
        )
        client = self.create_client({"EvalFile": str(network_path), "MultiPV": "3", "Policy Simulations": "8",})
        _, transcript, _ = client.search_position(
            "position startpos",
            "go depth 2 nodes 10000",
            10.0,
        )
        reports = parse_variations(transcript)
        self.assertEqual(
            len(reports),
            3,
        )
        roots = {report["moves"][0] for report in reports}
        self.assertEqual(
            len(roots),
            3,
        )

        for root_move in roots:
            self.assertRegex(
                root_move,
                r"^(?:[a-h]2[a-h][34]|b1[ac]3|g1[fh]3)$",
            )

        for line in transcript:
            node_match = re.search(
                r" nodes (\d+)(?: |$)",
                line,
            )

            if node_match:
                self.assertLessEqual(
                    int(node_match[1]),
                    10000,
                )

    def test_corrupt_network_replacement_preserves_previous_evaluation(self) -> None:
        network_path = train_fixture_network(
            self.engine_path,
            self.root,
        )
        corrupt_path = self.root / "corrupt.nnue"
        payload = bytearray(network_path.read_bytes())
        payload[100] ^= 1
        corrupt_path.write_bytes(payload)
        client = self.create_client({"EvalFile": str(network_path), "Policy Simulations": "0",})
        _, before, _ = client.search_position(
            "position startpos",
            "go searchmoves e2e4 depth 1",
            10.0,
        )
        client.send_command(f"setoption name EvalFile value {corrupt_path}")
        client.send_command("setoption name Clear Hash")
        client.send_command("isready")
        rejected = client.wait_for_line(
            "readyok",
            10.0,
        )
        self.assertTrue(any("checksum" in line for line in rejected))
        _, after, _ = client.search_position(
            "position startpos",
            "go searchmoves e2e4 depth 1",
            10.0,
        )
        self.assertEqual(
            parse_variations(before)[0]["score"],
            parse_variations(after)[0]["score"],
        )

    def test_network_replacement_cancels_active_search_and_remains_usable(self) -> None:
        network_path = train_fixture_network(
            self.engine_path,
            self.root,
        )
        client = self.create_client({"Policy Simulations": "0",})
        client.send_command("position startpos")
        client.is_searching = True
        client.send_command("go infinite")
        client.wait_for_line(
            "info depth 1 ",
            10.0,
        )
        client.send_command(f"setoption name EvalFile value {network_path}")
        client.send_command("isready")
        transcript = client.wait_for_line(
            "readyok",
            10.0,
        )
        client.is_searching = False
        self.assertEqual(
            len([line for line in transcript if line.startswith("bestmove ")]),
            1,
        )
        self.assertTrue(any("Loaded network fingerprint" in line for line in transcript))
        best_move, _, _ = client.search_position(
            "position startpos",
            "go searchmoves d2d4 depth 1",
            10.0,
        )
        self.assertEqual(
            best_move,
            "d2d4",
        )

    def test_special_move_search_restrictions_work_with_trained_weights(self) -> None:
        network_path = train_fixture_network(
            self.engine_path,
            self.root,
        )
        client = self.create_client({"EvalFile": str(network_path), "Policy Simulations": "8",})

        for fen, move_name in [("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1",), ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6",), ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n",),]:
            client.start_game()
            best_move, _, _ = client.search_position(
                f"position fen {fen}",
                f"go searchmoves {move_name} depth 1",
                10.0,
            )
            self.assertEqual(
                best_move,
                move_name,
            )

    def test_eof_during_active_search_stops_worker_and_exits_cleanly(self) -> None:
        completed = subprocess.run(
            [str(self.engine_path),],
            input="uci\nposition startpos\ngo infinite\n",
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr,
        )
        self.assertEqual(
            len([line for line in completed.stdout.splitlines() if line.startswith("bestmove ")]),
            1,
        )

    def test_invalid_utf8_command_is_rejected_without_losing_readiness(self) -> None:
        completed = subprocess.run(
            [str(self.engine_path),],
            input=b"uci\n\xff\n\r\nisready\r\nquit\n",
            capture_output=True,
            timeout=10.0,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr,
        )
        self.assertIn(
            b"readyok",
            completed.stdout,
        )
        self.assertIn(
            b"UTF-8",
            completed.stdout,
        )

    def test_invalid_search_arguments_do_not_poison_following_searches(self) -> None:
        client = self.create_client({"Policy Simulations": "0",})

        for command in ["go depth", "go nodes 0", "go searchmoves", "go searchmoves e2e5", "go movestogo 0", "go unsupported 1",]:
            client.send_command(command)
            client.send_command("isready")
            transcript = client.wait_for_line(
                "readyok",
                10.0,
            )
            self.assertTrue(any(line.startswith("info string Failed ") for line in transcript))
            self.assertFalse(any(line.startswith("bestmove ") for line in transcript))

        best_move, _, _ = client.search_position(
            "position startpos",
            "go searchmoves e2e4 depth 1",
            10.0,
        )
        self.assertEqual(
            best_move,
            "e2e4",
        )

    def test_excessive_position_history_preserves_previous_root(self) -> None:
        client = self.create_client({"Policy Simulations": "0",})
        client.send_command("position startpos moves e2e4")
        moves = " ".join(["g1f3", "g8f6", "f3g1", "f6g8",] * 1000)
        client.send_command("position startpos moves " + moves)
        client.send_command("isready")
        transcript = client.wait_for_line(
            "readyok",
            10.0,
        )
        self.assertTrue(any("move history exceeds" in line for line in transcript))
        client.send_command("go searchmoves e7e5 depth 1")
        transcript = client.wait_for_line(
            "bestmove ",
            10.0,
        )
        self.assertEqual(
            transcript[-1].split()[1],
            "e7e5",
        )

class StockfishIntegrationTests(ProcessTests):
    @classmethod
    def setUpClass(cls) -> None:
        import chess

        cls.chess_version = chess.__version__

        if not cls.stockfish_path.is_file():
            raise FileNotFoundError("Failed to run Stockfish integration tests because its binary is missing.")

    def create_match_arguments(
        self,
        candidate: Path,
        output_path: Path,
    ) -> list[str]:
        return [
            sys.executable,
            str(PROJECT_DIRECTORY / "scripts/play_stockfish.py"),
            "--candidate", str(candidate),
            "--stockfish", str(self.stockfish_path),
            "--openings", str(PROJECT_DIRECTORY / "test/openings.epd"),
            "--output", str(output_path),
            "--pairs", "1",
            "--clock_milliseconds", "2000",
            "--increment_milliseconds", "20",
            "--maximum_plies", "8",
        ]

    def validate_match_artifacts(
        self,
        output_path: Path,
    ) -> MatchSummary:
        import chess.pgn

        manifest = json.loads((output_path / "manifest.json").read_text())
        self.assertEqual(
            manifest["python_chess_version"],
            self.chess_version,
        )
        summary = json.loads((output_path / "summary.json").read_text())
        self.assertTrue(summary["is_complete"])
        self.assertEqual(
            len(summary["games"]),
            2,
        )
        self.assertEqual(
            sum(summary["pentanomial_counts"]),
            1,
        )
        self.assertEqual(
            [game["candidate_color"] for game in summary["games"]],
            ["white", "black",],
        )

        for engine_name in ["candidate", "stockfish",]:
            engine = manifest[engine_name]
            self.assertEqual(
                hashlib.sha256(Path(engine["path"]).read_bytes()).hexdigest(),
                engine["sha256"],
            )

        game_index = 0

        with (output_path / "games.pgn").open() as source:
            while True:
                game = chess.pgn.read_game(source)

                if game is None:
                    break

                self.assertFalse(game.errors)
                board = game.board()
                ply_count = 0

                for chess_move in game.mainline_moves():
                    self.assertIn(
                        chess_move,
                        board.legal_moves,
                    )
                    board.push(chess_move)
                    ply_count += 1

                self.assertEqual(
                    game.headers["Result"],
                    summary["games"][game_index]["result"],
                )
                self.assertEqual(
                    ply_count,
                    summary["games"][game_index]["ply_count"],
                )
                game_index += 1

        self.assertEqual(
            game_index,
            2,
        )
        transcripts = [json.loads(line) for line in (output_path / "protocol.jsonl").read_text().splitlines()]
        self.assertTrue(transcripts)
        self.assertTrue(all(record["elapsed_milliseconds"] >= 0 for record in transcripts))

        return summary

    def test_actual_stockfish_clocked_pairs_retain_legal_pgn_and_provenance(self) -> None:
        output_path = self.root / "clocked_match"
        self.run_program(self.create_match_arguments(
            self.engine_path,
            output_path,
        ))
        summary = self.validate_match_artifacts(output_path)
        self.assertTrue(all(not game["has_forfeit"] for game in summary["games"]))
        self.assertTrue(all(game["ply_count"] == 8 for game in summary["games"]))
        self.assertTrue(all("ply limit" in game["termination"] for game in summary["games"]))

    def test_teacher_training_export_and_live_mate_match_pipeline(self) -> None:
        import chess

        pgn_path = self.root / "teacher.pgn"
        dataset_path = self.root / "teacher_positions.txt"
        pgn_path.write_text(TEACHER_PGN)
        self.run_program([
            sys.executable, str(PROJECT_DIRECTORY / "scripts/generate_training_data.py"),
            "--pgn", str(pgn_path), "--teacher", str(self.stockfish_path),
            "--output", str(dataset_path), "--nodes", "1000",
            "--sample_interval", "2", "--maximum_records", "100",
        ])
        manifest = json.loads(dataset_path.with_suffix(".manifest.json").read_text())
        records = dataset_path.read_text().splitlines()
        self.assertGreaterEqual(
            len(records),
            4,
        )
        self.assertEqual(
            manifest["record_count"],
            len(records),
        )
        self.assertEqual(
            manifest["dataset_sha256"],
            hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        )
        game_groups = {}
        grouped_records = []

        for record in records:
            fields = [field.strip() for field in record.split("|")]
            board = chess.Board(fields[0])
            self.assertTrue(board.is_valid())
            self.assertEqual(
                fields[2],
                "?",
            )
            total_probability = 0.0

            for target in fields[3].split():
                move_name, probability_text = target.split(":")
                self.assertIn(
                    chess.Move.from_uci(move_name),
                    board.legal_moves,
                )
                total_probability += float(probability_text)

            self.assertAlmostEqual(
                total_probability,
                1.0,
                places=6,
            )

            if fields[4] not in game_groups:
                game_groups[fields[4]] = "game_0" if not game_groups else "game_3"

            fields[4] = game_groups[fields[4]]
            grouped_records.append(" | ".join(fields))

        self.assertEqual(
            len(game_groups),
            2,
        )
        grouped_path = self.root / "grouped_positions.txt"
        grouped_path.write_text("\n".join(grouped_records) + "\n")
        network_path = self.root / "teacher_model.nnue"
        self.run_program([str(self.engine_path), "train_network", str(grouped_path), str(network_path), "2", "--batch_size", "4",])
        training_manifest = json.loads(network_path.with_suffix(".training.json").read_text())
        self.assertTrue(training_manifest["has_trained_policy"])
        self.assertFalse(training_manifest["has_trained_outcome"])
        self.assertGreater(
            training_manifest["validation_record_count"],
            0,
        )
        openings_path = self.root / "mate.epd"
        openings_path.write_text('7k/8/5KQ1/8/8/8/8/8 w - - id "Forced mate fixture";\n')
        output_path = self.root / "trained_match"
        arguments = self.create_match_arguments(
            self.engine_path,
            output_path,
        )
        arguments[arguments.index("--openings") + 1] = str(openings_path)
        arguments += ["--candidate_option", f"EvalFile={network_path}",]
        self.run_program(arguments)
        summary = self.validate_match_artifacts(output_path)
        self.assertEqual(
            (summary["wins"], summary["draws"], summary["losses"],),
            (1, 0, 1,),
        )
        self.assertTrue(all(game["termination"] == "Checkmate." for game in summary["games"]))
        match_manifest = json.loads((output_path / "manifest.json").read_text())
        self.assertEqual(
            match_manifest["candidate"]["network_sha256"],
            hashlib.sha256(network_path.read_bytes()).hexdigest(),
        )

    def test_match_runner_records_illegal_moves_as_forfeits(self) -> None:
        candidate_path = create_uci_fixture(self.root)
        output_path = self.root / "illegal_match"
        arguments = self.create_match_arguments(
            candidate_path,
            output_path,
        ) + ["--candidate_option", "best_move=e2e5",]
        self.run_program(arguments)
        summary = self.validate_match_artifacts(output_path)
        self.assertEqual(
            summary["losses"],
            2,
        )
        self.assertTrue(all(game["has_forfeit"] for game in summary["games"]))
        self.assertTrue(all("Illegal move" in game["termination"] for game in summary["games"]))

    def test_crashed_engine_leaves_explicitly_incomplete_match_results(self) -> None:
        candidate_path = create_uci_fixture(self.root)
        output_path = self.root / "crashed_match"
        arguments = self.create_match_arguments(
            candidate_path,
            output_path,
        ) + ["--candidate_option", "fault_mode=crash",]
        self.run_program(
            arguments,
            False,
        )
        summary = json.loads((output_path / "summary.json").read_text())
        self.assertFalse(summary["is_complete"])
        self.assertEqual(
            len(summary["games"]),
            1,
        )
        self.assertTrue(summary["games"][0]["has_forfeit"])

    def test_match_runner_records_clock_overruns_as_time_forfeits(self) -> None:
        candidate_path = create_uci_fixture(self.root)
        output_path = self.root / "time_forfeit_match"
        arguments = self.create_match_arguments(
            candidate_path,
            output_path,
        ) + ["--candidate_option", "delay_milliseconds=250",]
        arguments[arguments.index("--clock_milliseconds") + 1] = "100"
        arguments[arguments.index("--increment_milliseconds") + 1] = "0"
        self.run_program(arguments)
        summary = self.validate_match_artifacts(output_path)
        self.assertEqual(
            summary["losses"],
            2,
        )
        self.assertTrue(all(game["has_forfeit"] for game in summary["games"]))
        self.assertTrue(all(game["termination"] == "Time forfeit." for game in summary["games"]))

    def test_match_runner_refuses_existing_results_and_invalid_openings(self) -> None:
        output_path = self.root / "retained_match"
        output_path.mkdir()
        marker_path = output_path / "retained.txt"
        marker_path.write_text("Preserved match data.\n")
        self.run_program(
            self.create_match_arguments(
                self.engine_path,
                output_path,
            ),
            False,
        )
        self.assertEqual(
            marker_path.read_text(),
            "Preserved match data.\n",
        )
        openings_path = self.root / "invalid.epd"
        openings_path.write_text("8/8/8/8/8/8/8/8 w - -\n")
        output_path = self.root / "invalid_match"
        arguments = self.create_match_arguments(
            self.engine_path,
            output_path,
        )
        arguments[arguments.index("--openings") + 1] = str(openings_path)
        self.run_program(
            arguments,
            False,
        )
        self.assertFalse(output_path.exists())

    def test_teacher_rejects_existing_outputs_and_unsupported_options(self) -> None:
        pgn_path = self.root / "teacher.pgn"
        pgn_path.write_text(TEACHER_PGN)
        output_path = self.root / "positions.txt"
        arguments = [sys.executable, str(PROJECT_DIRECTORY / "scripts/generate_training_data.py"), "--pgn", str(pgn_path), "--teacher", str(self.stockfish_path), "--output", str(output_path),]
        output_path.write_text("Preserved labels.\n")
        self.run_program(
            arguments,
            False,
        )
        self.assertEqual(
            output_path.read_text(),
            "Preserved labels.\n",
        )
        output_path.unlink()
        self.run_program(
            arguments + ["--teacher_option", "Unsupported option=1",],
            False,
        )
        self.assertFalse(output_path.exists())

    def test_reference_perft_matches_independent_python_chess_positions(self) -> None:
        completed = self.run_program([
            sys.executable, str(PROJECT_DIRECTORY / "scripts/verify_perft.py"),
            str(self.engine_path), "--positions", "128",
        ])
        self.assertIn(
            "Verified 128",
            completed.stdout,
        )

def run_integration_tests() -> None:
    parser = argparse.ArgumentParser(description="Run engine integration tests, with optional real Stockfish and python-chess workflows.")
    parser.add_argument(
        "engine_path",
        type=Path,
    )
    parser.add_argument(
        "--stockfish",
        type=Path,
    )
    arguments = parser.parse_args()
    ProcessTests.engine_path = arguments.engine_path.resolve()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(EngineIntegrationTests)

    if arguments.stockfish is not None:
        ProcessTests.stockfish_path = arguments.stockfish.resolve()
        suite.addTests(loader.loadTestsFromTestCase(StockfishIntegrationTests))

    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if outcome.wasSuccessful() else 1)

if __name__ == "__main__":
    run_integration_tests()
