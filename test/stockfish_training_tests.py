import argparse
import json
import math
import random
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict

from test_support import (
    PROJECT_DIRECTORY,
    TRAINING_RECORDS,
)

sys.path.insert(
    0,
    str(PROJECT_DIRECTORY / "scripts",),
)

from uci_client import UciClient

START_FEN = TRAINING_RECORDS.split("|",)[0].strip()
VERSION = 0x7af32f20
SMALL_HASH = 0x1c103c92
TRANSFORMER_HASH = 0x7f234db8
STACK_HASH = 0x6333712a
LEB_MAGIC = b"COMPRESSED_LEB128"
BUCKET_BYTES = 3304
ENGINE_PATH = PROJECT_DIRECTORY / "build/release/alekhine"
STOCKFISH_PATH = Path("/usr/bin/stockfish",)

class BucketScores(TypedDict,):
    bucket_index: int
    psqt_score: int
    positional_score: int
    centipawn_score: int

class NetworkInspection(TypedDict,):
    hidden_width: int
    network_fingerprint: int
    bucket_index: int
    centipawn_scale: float
    buckets: list[BucketScores]

def encode_signed(parameter: int,) -> bytes:
    encoded = bytearray()

    while True:
        byte = parameter & 127
        parameter >>= 7
        is_finished = ((parameter == 0) and ((byte & 64) == 0)) or ((parameter == -1) and ((byte & 64) != 0))
        encoded.append(byte if is_finished else byte | 128,)

        if is_finished:
            return bytes(encoded,)

def encode_block(payload: bytes,) -> bytes:
    return LEB_MAGIC + struct.pack(
        "<I",
        len(payload,),
    ) + payload

def create_fixture_bytes() -> bytes:
    description = b"Original synthetic format and optimizer fixture."

    header = struct.pack(
        "<III",
        VERSION,
        SMALL_HASH,
        len(description,),
    ) + description + struct.pack(
        "<I",
        TRANSFORMER_HASH,
    )

    transformer = encode_block(encode_signed(100,) * 128,) + encode_block(bytes(22528 * 128,),) + encode_block(bytes(22528 * 8,),)

    stack = struct.pack(
        "<I",
        STACK_HASH,
    ) + bytes(16 * 4 + 16 * 128 + 32 * 4 + 32 * 32 + 4 + 32,)

    return header + transformer + stack * 8

def create_dataset() -> str:
    return "\n".join((" | ".join([
        fields[0].replace(
            " b ",
            " w ",
        ),
        "1000",
        "1",
        "?",
        fields[4].strip(),
    ],)
        for line in TRAINING_RECORDS.splitlines()
        if (fields := line.split("|",))),) + "\n"

class NetworkProcessTests(unittest.TestCase,):
    def setUp(self,) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="alekhine_stockfish_training_",)
        self.addCleanup(self.directory.cleanup,)
        self.root = Path(self.directory.name,)
        self.source_path = self.root / "source.nnue"
        self.source_path.write_bytes(create_fixture_bytes(),)
        self.dataset_path = self.root / "positions.txt"
        self.dataset_path.write_text(create_dataset(),)

    def run_engine(
        self,
        arguments: list[str],
        is_success_expected: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [str(ENGINE_PATH,),] + arguments,
            capture_output=True,
            text=True,
            timeout=60.0,
        )

        self.assertEqual(
            completed.returncode == 0,
            is_success_expected,
            completed.stdout + completed.stderr,
        )

        return completed

    def train_model(
        self,
        name: str,
        options: list[str],
    ) -> Path:
        output_path = self.root / name

        settings = {
            "--batch_size": "2",
            "--learning_rate": "0.01",
            "--seed": "42",
        }

        settings.update(dict(zip(
            options[::2],
            options[1::2],
        ),),)

        configured_options = [part for key, option in settings.items() for part in [key, option,]]

        self.run_engine([
            "fine_tune_stockfish",
            str(self.source_path,),
            str(self.dataset_path,),
            str(output_path,),
            "20",
        ] + configured_options,)

        return output_path

    def inspect_model(
        self,
        path: Path,
    ) -> NetworkInspection:
        return json.loads(self.run_engine(["inspect_stockfish_network", str(path,),],).stdout,)

class StockfishFormatTests(NetworkProcessTests,):
    def test_synthetic_format_has_exact_integer_zero_scores(self,) -> None:
        report = self.inspect_model(self.source_path,)

        self.assertEqual(
            report["hidden_width"],
            128,
        )

        self.assertEqual(
            [bucket["centipawn_score"] for bucket in report["buckets"]],
            [0,] * 8,
        )

    def test_fine_tuning_changes_dense_weights_and_preserves_frozen_bytes(self,) -> None:
        output_path = self.train_model(
            "trained.nnue",
            [],
        )

        source = self.source_path.read_bytes()
        trained = output_path.read_bytes()
        frozen_end = len(source,) - 8 * BUCKET_BYTES

        self.assertEqual(
            trained[:frozen_end],
            source[:frozen_end],
        )

        self.assertEqual(
            trained[frozen_end:frozen_end + 7 * BUCKET_BYTES],
            source[frozen_end:frozen_end + 7 * BUCKET_BYTES],
        )

        self.assertNotEqual(
            trained,
            source,
        )

        manifest = json.loads(output_path.with_suffix(".training.json",).read_text(),)
        self.assertTrue(manifest["is_feature_transformer_frozen"],)
        self.assertTrue(manifest["are_psqt_weights_frozen"],)
        self.assertFalse(manifest["are_policy_targets_used"],)

        self.assertGreater(
            manifest["changed_parameter_count"],
            0,
        )

        self.assertLess(
            manifest["best_validation_loss"],
            manifest["initial_validation_loss"],
        )

        self.assertEqual(
            manifest["network_fingerprint"],
            self.inspect_model(output_path,)["network_fingerprint"],
        )

    def test_export_is_deterministic_and_preserves_existing_outputs(self,) -> None:
        first_path = self.train_model(
            "first.nnue",
            [],
        )

        second_path = self.train_model(
            "second.nnue",
            [],
        )

        self.assertEqual(
            first_path.read_bytes(),
            second_path.read_bytes(),
        )

        original = first_path.read_bytes()

        self.run_engine(
            ["fine_tune_stockfish", str(self.source_path,), str(self.dataset_path,), str(first_path,),],
            False,
        )

        self.assertEqual(
            first_path.read_bytes(),
            original,
        )

    def test_quantized_baseline_is_retained_when_updates_do_not_change_scores(self,) -> None:
        output_path = self.train_model(
            "unchanged.nnue",
            ["--learning_rate", "0.000000001",],
        )

        self.assertEqual(
            output_path.read_bytes(),
            self.source_path.read_bytes(),
        )

        manifest = json.loads(output_path.with_suffix(".training.json",).read_text(),)

        self.assertEqual(
            manifest["best_epoch"],
            0,
        )

    def test_validation_labels_do_not_change_training_gradients(self,) -> None:
        first_path = self.root / "first.nnue"
        arguments = ["fine_tune_stockfish", str(self.source_path,), str(self.dataset_path,), str(first_path,), "1", "--learning_rate", "0.01",]
        first = self.run_engine(arguments,)

        self.dataset_path.write_text(create_dataset().replace(
            "1000 | 1 | ? | game_3",
            "-1000 | 0 | ? | game_3",
        ),)

        arguments[3] = str(self.root / "second.nnue",)
        second = self.run_engine(arguments,)

        self.assertEqual(
            re.search(
                r"training loss ([0-9.]+)",
                first.stdout,
            ).group(1,),
            re.search(
                r"training loss ([0-9.]+)",
                second.stdout,
            ).group(1,),
        )

        first_manifest = json.loads(first_path.with_suffix(".training.json",).read_text(),)
        second_manifest = json.loads((self.root / "second.training.json").read_text(),)

        self.assertGreater(
            first_manifest["best_epoch"],
            0,
        )

        self.assertEqual(
            second_manifest["best_epoch"],
            0,
        )

    def test_dataset_deduplicates_before_splitting(self,) -> None:
        first_line = create_dataset().splitlines()[0]

        self.dataset_path.write_text(create_dataset() + first_line.replace(
            "0 1 |",
            "1 2 |",
        ).replace(
            "game_0",
            "game_3",
        ) + "\n",)

        output_path = self.train_model(
            "deduplicated.nnue",
            [],
        )

        manifest = json.loads(output_path.with_suffix(".training.json",).read_text(),)

        self.assertEqual(
            [manifest["training_record_count"], manifest["validation_record_count"], manifest["duplicate_record_count"],],
            [2, 1, 1,],
        )

    def test_decoder_rejects_wrong_architecture_truncation_and_trailing_bytes(self,) -> None:
        original = self.source_path.read_bytes()
        invalid_architecture = bytearray(original,)
        invalid_architecture[4:8] = bytes(4,)

        for invalid in [original[:100], original[:-1], original + b"x", invalid_architecture,]:
            self.source_path.write_bytes(invalid,)

            self.run_engine(
                ["inspect_stockfish_network", str(self.source_path,),],
                False,
            )

    def test_invalid_training_inputs_do_not_create_outputs(self,) -> None:
        for options in [["--learning_rate", "nan",], ["--batch_size", "0",], ["--maximum_record_count", "2",],]:
            output_path = self.root / "rejected.nnue"

            self.run_engine(
                ["fine_tune_stockfish", str(self.source_path,), str(self.dataset_path,), str(output_path,),] + options,
                False,
            )

            self.assertFalse(output_path.exists(),)
            self.assertFalse(output_path.with_suffix(".training.json",).exists(),)

        self.dataset_path.write_text("Invalid FEN | 0 | - | - | game\n",)

        self.run_engine(
            ["fine_tune_stockfish", str(self.source_path,), str(self.dataset_path,), str(output_path,),],
            False,
        )

        self.assertFalse(output_path.exists(),)

    def test_stockfish_model_loads_through_uci_and_survives_weight_resume(self,) -> None:
        output_path = self.train_model(
            "trained.nnue",
            [],
        )

        client = UciClient(
            ENGINE_PATH,
            {"EvalFile": str(output_path,),},
        )

        self.addCleanup(client.close_process,)
        client.send_command("position startpos",)
        client.send_command("go nodes 32",)

        transcript = client.wait_for_line(
            "bestmove",
            20.0,
        )

        self.assertRegex(
            transcript[-1],
            r"^bestmove [a-h][1-8][a-h][1-8]",
        )

        self.source_path = output_path

        export_path = self.train_model(
            "resumed.nnue",
            ["--learning_rate", "0.000000001",],
        )

        self.assertEqual(
            export_path.read_bytes(),
            output_path.read_bytes(),
        )

class RealStockfishTests(NetworkProcessTests,):
    @classmethod
    def setUpClass(cls,) -> None:
        cls.network_directory = tempfile.TemporaryDirectory(prefix="alekhine_real_stockfish_weights_",)
        cls.addClassCleanup(cls.network_directory.cleanup,)
        cls.big_path = Path(cls.network_directory.name,) / "big.nnue"
        cls.small_path = Path(cls.network_directory.name,) / "small.nnue"

        completed = subprocess.run(
            [str(STOCKFISH_PATH,),],
            input=f"export_net {cls.big_path} {cls.small_path}\nquit\n",
            capture_output=True,
            text=True,
            timeout=30.0,
        )

        if (completed.returncode != 0) or not cls.big_path.is_file() or not cls.small_path.is_file():
            raise RuntimeError(f"Failed to export Stockfish 18 weights: {completed.stdout} {completed.stderr}",)

    def test_real_networks_reexport_byte_identically(self,) -> None:
        for source_path in [self.big_path, self.small_path,]:
            client = UciClient(
                ENGINE_PATH,
                {"EvalFile": str(source_path,),},
            )

            self.addCleanup(client.close_process,)
            self.source_path = source_path

            output_path = self.train_model(
                source_path.name,
                ["--learning_rate", "0.000000001",],
            )

            self.assertEqual(
                output_path.read_bytes(),
                source_path.read_bytes(),
            )

    def test_big_evaluation_matches_stockfish_across_legal_positions(self,) -> None:
        import chess

        board = chess.Board()
        random_sequence = random.Random(42,)
        positions = [board.fen(),]

        for ply in range(180,):
            if board.is_game_over():
                break

            board.push(random_sequence.choice(list(board.legal_moves,),),)

            if (ply % 3 == 0) and not board.is_check():
                positions.append(board.fen(),)

        positions += [
            "7k/8/8/8/8/8/8/K7 w - - 0 1",
            "7k/8/8/8/8/8/P7/K7 w - - 0 1",
            "7k/8/8/8/8/8/PPPP4/K7 w - - 0 1",
            "7k/8/8/8/8/NNNN4/PPPP4/K7 w - - 0 1",
        ]

        original_pieces = chess.Board().piece_map()
        nonking_squares = sorted((square for square, piece in original_pieces.items() if piece.piece_type != chess.KING),)

        for piece_count in [2, 5, 9, 13, 17, 21, 25, 29,]:
            phase_board = chess.Board.empty()

            phase_board.set_piece_at(
                chess.E1,
                original_pieces[chess.E1],
            )

            phase_board.set_piece_at(
                chess.E8,
                original_pieces[chess.E8],
            )

            for square in nonking_squares[:piece_count - 2]:
                phase_board.set_piece_at(
                    square,
                    original_pieces[square],
                )

            self.assertTrue(phase_board.is_valid(),)
            positions.append(phase_board.fen(),)

        self.assertGreaterEqual(
            len(positions,),
            32,
        )

        client = UciClient(
            STOCKFISH_PATH,
            {"EvalFile": str(self.big_path,), "EvalFileSmall": str(self.small_path,),},
        )

        self.addCleanup(client.close_process,)

        for fen in positions:
            completed = self.run_engine(["inspect_stockfish_network", str(self.big_path,), fen,],)
            report = json.loads(completed.stdout,)
            client.send_command(f"position fen {fen}",)
            client.send_command("eval",)
            client.send_command("isready",)

            transcript = client.wait_for_line(
                "readyok",
                20.0,
            )

            rows = [
                line.split("|",)[1:5]
                for line in transcript
                if re.match(
                    r"^\|\s+[0-7]\s+\|",
                    line,
                )
            ]

            self.assertEqual(
                len(rows,),
                8,
                fen,
            )

            for row, bucket in zip(
                rows,
                report["buckets"],
            ):
                expected_scores = [
                    round(float(field.replace(
                        " ",
                        "",
                    ),) * 100,)
                    for field in row[1:]
                ]

                actual_scores = [
                    round(bucket["psqt_score"] * report["centipawn_scale"],),
                    round(bucket["positional_score"] * report["centipawn_scale"],),
                    bucket["centipawn_score"],
                ]

                for expected, actual in zip(
                    expected_scores,
                    actual_scores,
                ):
                    self.assertLessEqual(
                        abs(expected - actual,),
                        1,
                        f"Evaluation differs for {fen}, bucket {row[0]}: {actual_scores} versus {expected_scores}.",
                    )

    def test_modified_small_network_loads_and_searches_in_stockfish(self,) -> None:
        self.source_path = self.small_path

        output_path = self.train_model(
            "modified_small.nnue",
            [],
        )

        manifest = json.loads(output_path.with_suffix(".training.json",).read_text(),)

        self.assertGreater(
            manifest["changed_parameter_count"],
            0,
        )

        client = UciClient(
            STOCKFISH_PATH,
            {"EvalFileSmall": str(output_path,),},
        )

        self.addCleanup(client.close_process,)
        client.send_command("position fen 7k/8/8/8/8/8/Q1Q5/K7 w - - 0 1",)
        client.send_command("go nodes 64",)

        transcript = client.wait_for_line(
            "bestmove",
            20.0,
        )

        self.assertRegex(
            transcript[-1],
            r"^bestmove [a-h][1-8][a-h][1-8]",
        )

    def test_small_evaluation_matches_stockfish_in_material_imbalance_positions(self,) -> None:
        import chess

        client = UciClient(
            STOCKFISH_PATH,
            {"EvalFile": str(self.big_path,), "EvalFileSmall": str(self.small_path,),},
        )

        self.addCleanup(client.close_process,)

        positions = [
            "7k/8/8/8/8/8/Q1Q5/K7 w - - 0 1",
            "7k/8/8/8/8/8/Q1Q5/K7 b - - 0 1",
            "7k/8/8/8/8/8/Q1Q5/K7 w - - 50 1",
            "7k/5p2/8/8/8/8/R1R5/K7 w - - 0 1",
        ]

        nonpawn_weights = {
            chess.KNIGHT: 781,
            chess.BISHOP: 825,
            chess.ROOK: 1276,
            chess.QUEEN: 2538,
        }

        for fen in positions:
            board = chess.Board(fen,)
            self.assertTrue(board.is_valid(),)
            report = json.loads(self.run_engine(["inspect_stockfish_network", str(self.small_path,), fen,],).stdout,)
            scores = report["buckets"][report["bucket_index"]]
            blended_score = math.trunc((125 * scores["psqt_score"] + 131 * scores["positional_score"]) / 128,)

            self.assertGreater(
                abs(blended_score,),
                277,
            )

            complexity = abs(scores["psqt_score"] - scores["positional_score"],)
            corrected_score = blended_score - math.trunc(blended_score * complexity / 18236,)

            nonpawn_material = sum((weight * len(board.pieces(
                piece_type,
                color,
            ),)
                for piece_type, weight in nonpawn_weights.items()
                for color in [chess.WHITE, chess.BLACK,]),)

            pawn_count = sum((len(board.pieces(
                chess.PAWN,
                color,
            ),)
                for color in [chess.WHITE, chess.BLACK,]),)

            scaled_score = math.trunc(corrected_score * (77871 + nonpawn_material + 534 * pawn_count) / 77871,)
            damped_score = scaled_score - math.trunc(scaled_score * board.halfmove_clock / 199,)
            side_sign = 1 if board.turn == chess.WHITE else -1
            expected_score = round(damped_score * report["centipawn_scale"] * side_sign,)
            client.send_command(f"position fen {fen}",)
            client.send_command("eval",)
            client.send_command("isready",)

            transcript = client.wait_for_line(
                "readyok",
                20.0,
            )

            final_line = next((line for line in transcript if line.startswith("Final evaluation ",)),)
            actual_score = round(float(final_line.split()[2],) * 100,)

            self.assertEqual(
                expected_score,
                actual_score,
                fen,
            )

    def test_modified_big_network_loads_and_searches_in_stockfish_and_alekhine(self,) -> None:
        self.source_path = self.big_path

        output_path = self.train_model(
            "modified_big.nnue",
            [],
        )

        manifest = json.loads(output_path.with_suffix(".training.json",).read_text(),)

        self.assertGreater(
            manifest["changed_parameter_count"],
            0,
        )

        for executable in [STOCKFISH_PATH, ENGINE_PATH,]:
            client = UciClient(
                executable,
                {"EvalFile": str(output_path,),},
            )

            self.addCleanup(client.close_process,)
            client.send_command("position startpos",)
            client.send_command("go nodes 64",)

            transcript = client.wait_for_line(
                "bestmove",
                20.0,
            )

            self.assertRegex(
                transcript[-1],
                r"^bestmove [a-h][1-8][a-h][1-8]",
            )

def run_tests() -> None:
    global ENGINE_PATH, STOCKFISH_PATH
    parser = argparse.ArgumentParser(description="Verify original Stockfish-format loading, fine-tuning, and export.",)

    parser.add_argument(
        "engine_path",
        type=Path,
    )

    parser.add_argument(
        "--stockfish",
        type=Path,
    )

    arguments = parser.parse_args()
    ENGINE_PATH = arguments.engine_path.resolve()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(StockfishFormatTests,)

    if arguments.stockfish is not None:
        STOCKFISH_PATH = arguments.stockfish.resolve()
        suite.addTests(loader.loadTestsFromTestCase(RealStockfishTests,),)

    outcome = unittest.TextTestRunner(verbosity=2,).run(suite,)

    sys.exit(0 if outcome.wasSuccessful() else 1,)

if __name__ == "__main__":
    run_tests()
