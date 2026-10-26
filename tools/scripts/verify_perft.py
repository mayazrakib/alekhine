import argparse
import random
import re
import subprocess
from pathlib import Path

import chess

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
MOVE_PATTERN = re.compile(r"^([a-h][1-8][a-h][1-8][qrbn]?) (\d+)$",)

def count_reference_perft(
    board: chess.Board,
    depth_plies: int,
) -> int:
    if depth_plies == 0:
        return 1

    node_count = 0

    for move in board.legal_moves:
        board.push(move,)

        node_count += count_reference_perft(
            board,
            depth_plies - 1,
        )

        board.pop()

    return node_count

def verify_positions(arguments: argparse.Namespace,) -> None:
    generator = random.Random(arguments.seed,)
    board = chess.Board(START_FEN,)
    reference_node_count = 0

    for position_index in range(arguments.positions,):
        if (position_index % 160 == 0) or board.is_game_over():
            board = chess.Board(START_FEN if (position_index % 320 == 0) else KIWIPETE_FEN,)

        fen = board.fen(en_passant="fen",)

        engine_process = subprocess.run(
            [str(arguments.engine.resolve(),), "divide", str(arguments.depth_plies,), fen,],
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
        )

        engine_counts = {}

        for line in engine_process.stdout.splitlines():
            match = MOVE_PATTERN.fullmatch(line,)

            if match:
                engine_counts[match[1]] = int(match[2],)

        reference_counts = {}

        legal_moves = sorted(
            board.legal_moves,
            key=lambda move: move.uci(),
        )

        for move in legal_moves:
            board.push(move,)

            reference_counts[move.uci()] = count_reference_perft(
                board,
                arguments.depth_plies - 1,
            )

            board.pop()

        if engine_counts != reference_counts:
            raise AssertionError(f"Perft differs at position {position_index}: {fen}. Engine: {engine_counts}. Reference: {reference_counts}.",)

        reference_node_count += sum(reference_counts.values(),)
        board.push(generator.choice(legal_moves,),)

        if (position_index + 1) % 100 == 0:
            print(
                f"Verified {position_index + 1} positions.",
                flush=True,
            )

    print(f"Verified {arguments.positions} positions at depth {arguments.depth_plies}, covering {reference_node_count} leaf nodes with seed {arguments.seed}.",)

def run_verification() -> None:
    parser = argparse.ArgumentParser(description="Compare legal moves and perft against python-chess on deterministic random legal positions.",)

    parser.add_argument(
        "engine",
        type=Path,
    )

    parser.add_argument(
        "--positions",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--depth",
        dest="depth_plies",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261004,
    )

    arguments = parser.parse_args()

    if not (1 <= arguments.positions <= 100000) or not (1 <= arguments.depth_plies <= 4):
        parser.error("The position count must be between 1 and 100,000, and the depth must be between 1 and 4.",)

    try:
        verify_positions(arguments,)
    except (AssertionError, OSError, subprocess.SubprocessError,) as error:
        parser.exit(
            1,
            f"Failed to verify perft: {error}\n",
        )

if __name__ == "__main__":
    run_verification()
