#!/usr/bin/env python3
import sys
import time

def run_fixture() -> None:
    fault_mode = "normal"
    best_move = "e2e4"
    delay_milliseconds = 0
    is_pending = False

    for command in sys.stdin:
        command = command.strip()

        if command == "uci":
            print("id name Alekhine protocol test fixture",)
            print("option name Hash type spin default 16 min 1 max 512",)
            print("option name Threads type spin default 1 min 1 max 1",)
            print("option name fault_mode type string default normal",)
            print("option name best_move type string default e2e4",)
            print("option name delay_milliseconds type spin default 0 min 0 max 1000",)

            print(
                "uciok",
                flush=True,
            )
        elif command.startswith("setoption name fault_mode value ",):
            fault_mode = command.split(
                " value ",
                1,
            )[1]
        elif command.startswith("setoption name best_move value ",):
            best_move = command.split(
                " value ",
                1,
            )[1]
        elif command.startswith("setoption name delay_milliseconds value ",):
            delay_milliseconds = int(command.split(
                " value ",
                1,
            )[1],)
        elif command == "isready":
            if fault_mode == "invalid option":
                print("info string Failed to configure the test fixture.",)

            print(
                "readyok",
                flush=True,
            )
        elif command.startswith("go ",):
            if fault_mode == "crash":
                print(
                    "Forced fixture termination.",
                    file=sys.stderr,
                    flush=True,
                )

                raise SystemExit(1,)

            if fault_mode == "oversized":
                print(
                    "x" * 65537,
                    flush=True,
                )

                continue

            if fault_mode == "silent":
                is_pending = True

                continue

            if fault_mode == "malformed":
                print(
                    "bestmove",
                    flush=True,
                )

                continue

            if delay_milliseconds > 0:
                time.sleep(delay_milliseconds / 1000,)

            print(f"info depth 2 multipv 1 score cp 25 nodes 12 pv {best_move}",)

            print(
                f"bestmove {best_move}",
                flush=True,
            )
        elif command == "stop" and is_pending:
            print(
                f"bestmove {best_move}",
                flush=True,
            )

            is_pending = False
        elif command == "quit":
            return

if __name__ == "__main__":
    run_fixture()
