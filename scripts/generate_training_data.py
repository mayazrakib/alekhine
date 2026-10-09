import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

from uci_client import (
    UciClient,
    parse_options,
    parse_variations,
)

def compute_digest(path: Path,) -> str:
    with path.open("rb",) as source:
        return hashlib.file_digest(
            source,
            "sha256",
        ).hexdigest()

def generate_records() -> None:
    parser = argparse.ArgumentParser(description="Label quiet positions from PGN games with searched teacher scores and legal move distributions.",)
    parser.add_argument(
        "--pgn",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path(shutil.which("stockfish",) or "stockfish",),
    )
    parser.add_argument(
        "--teacher_option",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--maximum_records",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--policy_temperature_centipawns",
        type=float,
        default=100.0,
    )
    arguments = parser.parse_args()
    teacher = None

    try:
        import chess
        import chess.pgn

        if arguments.output.exists() or arguments.output.with_suffix(".manifest.json",).exists():
            raise ValueError("Failed to generate records because an output file already exists.",)

        if not (100 <= arguments.nodes <= 1000000000) or not (1 <= arguments.maximum_records <= 1000000):
            raise ValueError("Failed to generate records because the node or record budget is invalid.",)

        if not (1 <= arguments.sample_interval <= 100) or not math.isfinite(arguments.policy_temperature_centipawns,) or (arguments.policy_temperature_centipawns <= 0):
            raise ValueError("Failed to generate records because the sample interval or policy temperature is invalid.",)

        options = {"Threads": "1", "Hash": "16", "MultiPV": "3",}
        options.update(parse_options(arguments.teacher_option,),)
        teacher = UciClient(
            arguments.teacher,
            options,
        )
        pgn_digest = compute_digest(arguments.pgn,)
        teacher_digest = compute_digest(arguments.teacher,)
        seen_positions = set()
        record_count = 0
        game_count = 0
        arguments.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with arguments.pgn.open() as source, arguments.output.open("x",) as output:
            while record_count < arguments.maximum_records:
                game = chess.pgn.read_game(source,)

                if game is None:
                    break

                if game.errors:
                    raise ValueError(f"Failed to parse source game {game_count + 1}: {game.errors}",)

                game_id = f"{pgn_digest[:16]}_{game_count}"
                game_count += 1
                teacher.start_game()
                board = game.board()
                initial_fen = board.fen()
                moves = []
                white_outcome = {"1-0": "1", "0-1": "0", "1/2-1/2": "0.5",}.get(
                    game.headers.get("Result",),
                    "?",
                )
                termination = game.headers.get(
                    "Termination",
                    "",
                ).lower()

                if any((fragment in termination for fragment in (
                    "forfeit",
                    "failure",
                    "illegal",
                    "ply limit",
                )),):
                    white_outcome = "?"

                for ply, chess_move in enumerate(game.mainline_moves(),):
                    position_key = " ".join(board.fen().split()[:4],)
                    should_sample = (ply >= 8) and (ply % arguments.sample_interval == 0)
                    is_quiet = not board.is_check() and not board.is_capture(chess_move,) and not chess_move.promotion

                    if should_sample and is_quiet and not board.is_game_over(claim_draw=True,) and (position_key not in seen_positions):
                        position_command = f"position fen {initial_fen}"

                        if moves:
                            position_command += " moves " + " ".join(moves,)

                        best_move, transcript, elapsed_seconds = teacher.search_position(
                            position_command,
                            f"go nodes {arguments.nodes}",
                            60.0,
                        )
                        variations = parse_variations(transcript,)

                        if chess.Move.from_uci(best_move,) not in board.legal_moves:
                            raise ValueError("Failed to label a position because the teacher returned an illegal best move.",)

                        if variations and (variations[0]["moves"][0] == best_move) and (variations[0]["score_kind"] == "cp") and (abs(variations[0]["score"],) <= 20000):
                            selected = [variation for variation in variations if variation["score_kind"] == "cp"]
                            highest_score = max((variation["score"] for variation in selected),)
                            weighted_moves = {}

                            for variation in selected:
                                move_name = variation["moves"][0]

                                if chess.Move.from_uci(move_name,) not in board.legal_moves:
                                    raise ValueError("Failed to label a position because the teacher returned an illegal move.",)

                                weighted_moves[move_name] = math.exp((variation["score"] - highest_score) / arguments.policy_temperature_centipawns,)

                            probability_sum = sum(weighted_moves.values(),)
                            policy = " ".join((f"{move_name}:{weight / probability_sum:.8f}" for move_name, weight in weighted_moves.items()),)
                            white_score = variations[0]["score"] if board.turn == chess.WHITE else -variations[0]["score"]
                            output.write(f"{board.fen()} | {white_score} | {white_outcome} | {policy} | {game_id}\n",)
                            seen_positions.add(position_key,)
                            record_count += 1

                    board.push(chess_move,)
                    moves.append(chess_move.uci(),)

                    if record_count >= arguments.maximum_records:
                        break

                print(
                    f"Processed source game {game_count}; generated {record_count} records.",
                    flush=True,
                )

        if record_count == 0:
            raise ValueError("Failed to generate records because no suitable positions were found.",)

        if teacher_digest != compute_digest(arguments.teacher,):
            raise ValueError("Failed to verify the labels because the teacher binary changed during generation.",)

        manifest = {
            "source_pgn_sha256": pgn_digest,
            "dataset_sha256": compute_digest(arguments.output,),
            "teacher_identity": teacher.identity,
            "teacher_sha256": teacher_digest,
            "teacher_options": options,
            "search_node_count": arguments.nodes,
            "sample_interval": arguments.sample_interval,
            "policy_temperature_centipawns": arguments.policy_temperature_centipawns,
            "record_count": record_count,
            "game_count": game_count,
            "score_perspective": "white",
            "outcome_perspective": "white",
            "unknown_outcome": "?",
            "policy_target": "Softmax of searched MultiPV centipawn scores; unlisted moves receive zero target probability.",
            "python_chess_version": chess.__version__,
        }
        network_path = options.get("EvalFile",)

        if network_path and Path(network_path,).is_file():
            manifest["teacher_network_sha256"] = compute_digest(Path(network_path,),)

        arguments.output.with_suffix(".manifest.json",).write_text(json.dumps(
            manifest,
            indent=4,
        ) + "\n",)
        print(f"Generated {record_count} records in {arguments.output}.",)
    except ImportError as error:
        parser.exit(
            1,
            f"Failed to generate records because python-chess is required: {error}\n",
        )
    except (OSError, RuntimeError, ValueError, TimeoutError,) as error:
        parser.exit(
            1,
            f"Failed to generate records: {error}\n",
        )
    finally:
        if teacher is not None:
            teacher.close_process()

if __name__ == "__main__":
    generate_records()
