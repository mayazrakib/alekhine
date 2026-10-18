import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_support import TRAINING_RECORDS

sys.path.insert(
    0,
    str(Path(__file__,).resolve().parent.parent / "scripts",),
)

from uci_client import (
    UciClient,
    parse_options,
    parse_variations,
)

ENGINE_PATH = Path(sys.argv.pop(1,),).resolve()

class ResearchTests(unittest.TestCase,):
    def test_option_parser_rejects_duplicate_and_multiline_settings(self,) -> None:
        for settings in (["Hash=16", "Hash=32",], ["Hash=16\nquit",], ["Hash",], ["=16",],):
            with self.assertRaises(ValueError,):
                parse_options(settings,)

        self.assertEqual(
            parse_options(["EvalFile=/tmp/network with spaces.nnue",],),
            {"EvalFile": "/tmp/network with spaces.nnue",},
        )

    def test_variation_parser_keeps_exact_consistent_depths(self,) -> None:
        reports = parse_variations([
            "info depth 3 multipv 1 score cp 30 pv e2e4 e7e5",
            "info depth 3 multipv 2 score cp 20 pv d2d4 d7d5",
            "info depth 4 multipv 1 score cp 50 lowerbound pv g1f3",
            "info depth 4 multipv 2 score cp 10 pv b1c3",
            "info string score cp broken pv output",
        ],)

        self.assertEqual(
            [report["depth_plies"] for report in reports],
            [3, 3,],
        )

        self.assertEqual(
            [report["moves"][0] for report in reports],
            ["e2e4", "d2d4",],
        )

    def test_unsupported_options_fail_during_handshake(self,) -> None:
        with self.assertRaises(ValueError,):
            UciClient(
                ENGINE_PATH,
                {"Unsupported option": "1",},
            )

    def test_original_training_export_and_uci_loading(self,) -> None:
        with tempfile.TemporaryDirectory(prefix="alekhine_training_",) as directory:
            dataset_path = Path(directory,) / "positions.txt"
            output_path = Path(directory,) / "model.nnue"
            dataset_path.write_text(TRAINING_RECORDS,)

            completed = subprocess.run(
                [str(ENGINE_PATH,), "train_network", str(dataset_path,), str(output_path,), "2",],
                capture_output=True,
                text=True,
                timeout=20.0,
                check=True,
            )

            self.assertIn(
                "Saved network fingerprint",
                completed.stdout,
            )

            self.assertTrue(output_path.is_file(),)

            client = UciClient(
                ENGINE_PATH,
                {"EvalFile": str(output_path,), "Policy Simulations": "8", "policy_verification_depth_plies": "1",},
            )

            try:
                best_move, transcript, elapsed_seconds = client.search_position(
                    "position startpos",
                    "go searchmoves e2e4 depth 1",
                    10.0,
                )

                self.assertEqual(
                    best_move,
                    "e2e4",
                )

                self.assertTrue(any((line.startswith("info depth 1 ",) for line in transcript),),)

                self.assertLess(
                    elapsed_seconds,
                    10.0,
                )
            finally:
                client.close_process()

if __name__ == "__main__":
    unittest.main()
