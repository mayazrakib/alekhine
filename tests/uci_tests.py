import queue
import re
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

ENGINE_PATH = Path(sys.argv.pop(1,),).resolve()

class EngineProcess:
    def __init__(self,) -> None:
        self.process = subprocess.Popen(
            [str(ENGINE_PATH,),],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self.lines: queue.Queue[str | None] = queue.Queue()
        self.transcript: list[str] = []
        self.stderr_lines: list[str] = []

        self.output_thread = threading.Thread(
            target=self.read_output,
            daemon=True,
        )

        self.error_thread = threading.Thread(
            target=self.read_errors,
            daemon=True,
        )

        self.output_thread.start()
        self.error_thread.start()

    def read_output(self,) -> None:
        for line in self.process.stdout:
            self.lines.put(line.strip(),)

        self.lines.put(None,)

    def read_errors(self,) -> None:
        for line in self.process.stderr:
            self.stderr_lines.append(line.strip(),)

    def send_command(
        self,
        command: str,
    ) -> None:
        self.process.stdin.write(command + "\n",)
        self.process.stdin.flush()

    def wait_for_line(
        self,
        prefix: str,
    ) -> str:
        deadline_seconds = time.monotonic() + 15.0

        while True:
            remaining_seconds = deadline_seconds - time.monotonic()

            if remaining_seconds <= 0:
                raise AssertionError(f"Timed out waiting for {prefix}: {self.transcript}",)

            try:
                line = self.lines.get(timeout=remaining_seconds,)
            except queue.Empty as error:
                raise AssertionError(f"Timed out waiting for {prefix}.",) from error

            if line is None:
                raise AssertionError(f"The engine terminated: {self.stderr_lines}",)

            self.transcript.append(line,)

            if line.startswith(prefix,):
                return line

    def close_process(self,) -> None:
        if self.process.poll() is None:
            self.send_command("quit",)

        try:
            self.process.wait(timeout=10.0,)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

            raise AssertionError("The engine did not shut down after quit.",)
        finally:
            self.output_thread.join(timeout=5.0,)
            self.error_thread.join(timeout=5.0,)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process.stderr.close()

        if self.process.returncode != 0:
            raise AssertionError(f"The engine exited with {self.process.returncode}: {self.stderr_lines}",)

        if self.stderr_lines:
            raise AssertionError(f"The engine emitted unexpected diagnostics: {self.stderr_lines}",)

class UciTests(unittest.TestCase,):
    def setUp(self,) -> None:
        self.engine = EngineProcess()
        self.engine.send_command("uci",)
        self.engine.wait_for_line("uciok",)

    def tearDown(self,) -> None:
        self.engine.close_process()

    def test_identity_and_options(self,) -> None:
        self.assertIn(
            "id author Mayaz Rakib",
            self.engine.transcript,
        )

        self.assertTrue(any((line.startswith("id name alekhine 0.1.0",) for line in self.engine.transcript),),)
        self.assertTrue(any(("Threads type spin default 1 min 1 max 1" in line for line in self.engine.transcript),),)
        self.engine.send_command("isready",)
        self.engine.wait_for_line("readyok",)

    def test_fixed_depth_produces_legal_start_move(self,) -> None:
        self.engine.send_command("position startpos",)
        self.engine.send_command("go depth 2",)
        bestmove = self.engine.wait_for_line("bestmove ",).split()[1]

        self.assertRegex(
            bestmove,
            r"^(?:[a-h]2[a-h][34]|b1[ac]3|g1[fh]3)$",
        )

        self.assertTrue(any((line.startswith("info depth 2 ",) for line in self.engine.transcript),),)

    def test_node_limited_search(self,) -> None:
        self.engine.send_command("go nodes 3000",)
        self.engine.wait_for_line("bestmove ",)
        node_counts = []

        for line in self.engine.transcript:
            match = re.search(
                r" nodes (\d+) ",
                line,
            )

            if match:
                node_counts.append(int(match[1],),)

        self.assertTrue(node_counts,)

        self.assertLessEqual(
            max(node_counts,),
            3000,
        )

    def test_one_node_still_returns_a_move(self,) -> None:
        self.engine.send_command("go nodes 1",)
        bestmove = self.engine.wait_for_line("bestmove ",)

        self.assertNotIn(
            "bestmove 0000",
            bestmove,
        )

    def test_stop_interrupts_infinite_search(self,) -> None:
        self.engine.send_command("go infinite",)
        self.engine.wait_for_line("info depth 1 ",)
        self.engine.send_command("isready",)
        self.engine.wait_for_line("readyok",)
        self.assertFalse(any((line.startswith("bestmove ",) for line in self.engine.transcript),),)
        self.engine.send_command("stop",)
        self.engine.wait_for_line("bestmove ",)
        self.engine.send_command("stop",)
        self.engine.send_command("isready",)
        self.engine.wait_for_line("readyok",)

    def test_ponder_withholds_bestmove_until_ponderhit(self,) -> None:
        self.engine.send_command("go ponder depth 1",)
        self.engine.wait_for_line("info depth 1 ",)
        self.engine.send_command("isready",)
        self.engine.wait_for_line("readyok",)
        self.assertFalse(any((line.startswith("bestmove ",) for line in self.engine.transcript),),)
        self.engine.send_command("ponderhit",)
        self.engine.wait_for_line("bestmove ",)

    def test_stop_interrupts_completed_ponder(self,) -> None:
        self.engine.send_command("go ponder depth 1",)
        self.engine.wait_for_line("info depth 1 ",)
        self.engine.send_command("stop",)
        self.engine.wait_for_line("bestmove ",)

    def test_movetime_stops_search(self,) -> None:
        started_at_seconds = time.monotonic()
        self.engine.send_command("go movetime 150",)
        self.engine.wait_for_line("bestmove ",)

        self.assertLess(
            time.monotonic() - started_at_seconds,
            3.0,
        )

    def test_tournament_clock_and_increment(self,) -> None:
        self.engine.send_command("position startpos moves e2e4",)
        self.engine.send_command("go wtime 1000 btime 200 winc 100 binc 10 movestogo 10",)
        self.engine.wait_for_line("bestmove ",)

    def test_multipv_reports_distinct_root_moves(self,) -> None:
        self.engine.send_command("setoption name MultiPV value 3",)
        self.engine.send_command("go depth 2",)
        self.engine.wait_for_line("bestmove ",)
        reports = [line for line in self.engine.transcript if line.startswith("info depth 2 ",)]

        self.assertEqual(
            len(reports,),
            3,
        )

        roots = [line.split(" pv ",)[1].split()[0] for line in reports]

        self.assertEqual(
            len(set(roots,),),
            3,
        )

    def test_searchmoves_restricts_root(self,) -> None:
        self.engine.send_command("go searchmoves e2e4 depth 2",)
        self.assertTrue(self.engine.wait_for_line("bestmove ",).startswith("bestmove e2e4",),)

    def test_checkmate_returns_null_move(self,) -> None:
        self.engine.send_command("position fen 7k/6Q1/5K2/8/8/8/8/8 b - - 100 51",)
        self.engine.send_command("go depth 2",)

        self.assertEqual(
            self.engine.wait_for_line("bestmove ",),
            "bestmove 0000",
        )

    def test_mate_score_and_move(self,) -> None:
        self.engine.send_command("position fen 7k/8/5KQ1/8/8/8/8/8 w - - 0 1",)
        self.engine.send_command("go depth 2",)
        self.engine.wait_for_line("bestmove ",)
        self.assertTrue(any((" score mate 1 " in line for line in self.engine.transcript),),)

    def test_invalid_input_preserves_previous_position(self,) -> None:
        self.engine.send_command("position startpos",)
        self.engine.send_command("position startpos moves e2e5",)
        self.engine.wait_for_line("info string Failed to apply an illegal UCI move.",)
        self.engine.send_command("go searchmoves e2e4 depth 1",)
        self.assertTrue(self.engine.wait_for_line("bestmove ",).startswith("bestmove e2e4",),)

    def test_invalid_options_remain_recoverable(self,) -> None:
        for command in (
            "setoption name Hash value 1000000000000000000000",
            "setoption name Threads value 2",
            "setoption name EvalFile value /nonexistent/alekhine.network",
            "position fen 8/8/8/8/8/8/8/8 w - - 0 1",
            "go depth -2",
        ):
            self.engine.send_command(command,)
            self.engine.wait_for_line("info string ",)
            self.engine.send_command("isready",)
            self.engine.wait_for_line("readyok",)

        self.engine.send_command("go depth 1",)
        self.engine.wait_for_line("bestmove ",)

    def test_rapid_search_and_new_game_transitions(self,) -> None:
        for cycle_index in range(5,):
            self.engine.send_command("go infinite",)
            self.engine.send_command("position startpos moves d2d4",)
            self.engine.wait_for_line("bestmove ",)
            self.engine.send_command("ucinewgame",)
            self.engine.send_command("ucinewgame",)
            self.engine.send_command("isready",)
            self.engine.wait_for_line("readyok",)

        self.engine.send_command("go searchmoves e2e4 depth 1",)
        self.assertTrue(self.engine.wait_for_line("bestmove ",).startswith("bestmove e2e4",),)

    def test_oversized_command_is_rejected(self,) -> None:
        self.engine.send_command("x" * 65537,)
        self.engine.wait_for_line("info string Rejected a command exceeding 65,536 bytes.",)
        self.engine.send_command("isready",)
        self.engine.wait_for_line("readyok",)

    def test_hash_resize_and_bootstrap_network_reset(self,) -> None:
        self.engine.send_command("setoption name Hash value 1",)
        self.engine.send_command("setoption name Clear Hash",)
        self.engine.send_command("setoption name EvalFile value <bootstrap>",)
        line = self.engine.wait_for_line("info string Loaded network fingerprint ",)

        self.assertIn(
            "untrained bootstrap weights.",
            line,
        )

        self.engine.send_command("go depth 1",)
        self.engine.wait_for_line("bestmove ",)

    def test_hybrid_options_and_ablation_controls(self,) -> None:
        for command in (
            "setoption name Policy Simulations value 0",
            "setoption name policy_verification_depth_plies value 0",
            "setoption name policy_uncertainty_percent value 0",
            "setoption name aggression_margin_centipawns value 0",
            "setoption name is_selective_search_enabled value false",
        ):
            self.engine.send_command(command,)

        self.engine.send_command("isready",)
        self.engine.wait_for_line("readyok",)
        self.engine.send_command("go searchmoves e2e4 depth 2",)
        self.assertTrue(self.engine.wait_for_line("bestmove ",).startswith("bestmove e2e4",),)

        for command in (
            "setoption name policy_verification_depth_plies value 4",
            "setoption name policy_uncertainty_percent value 101",
            "setoption name aggression_margin_centipawns value -1",
            "setoption name is_selective_search_enabled value sometimes",
        ):
            self.engine.send_command(command,)
            self.engine.wait_for_line("info string Failed ",)

        self.engine.send_command("isready",)
        self.engine.wait_for_line("readyok",)

if __name__ == "__main__":
    unittest.main()
