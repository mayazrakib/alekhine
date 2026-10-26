import argparse
import hashlib
import json
import platform
import random
import shutil
import time
from pathlib import Path

from uci_client import (
    UciClient,
    parse_options,
    parse_variations,
)

PROJECT_DIRECTORY = Path(__file__,).resolve().parents[2]

def compute_digest(path: Path,) -> str:
    with path.open("rb",) as source:
        return hashlib.file_digest(
            source,
            "sha256",
        ).hexdigest()

def run_games() -> None:
    parser = argparse.ArgumentParser(description="Play paired, clocked games against Stockfish and retain PGN, protocol responses, and provenance.",)

    parser.add_argument(
        "--candidate",
        type=Path,
        default=PROJECT_DIRECTORY / "build/release/alekhine",
    )

    parser.add_argument(
        "--stockfish",
        type=Path,
        default=Path(shutil.which("stockfish",) or "stockfish",),
    )

    parser.add_argument(
        "--openings",
        type=Path,
        default=PROJECT_DIRECTORY / "tests/openings.epd",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--pairs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--clock_milliseconds",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--increment_milliseconds",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--maximum_plies",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261004,
    )

    parser.add_argument(
        "--candidate_option",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--stockfish_option",
        action="append",
        default=[],
    )

    arguments = parser.parse_args()
    clients = []

    try:
        import chess
        import chess.pgn

        if not (1 <= arguments.pairs <= 100000) or not (100 <= arguments.clock_milliseconds <= 86400000):
            raise ValueError("Failed to prepare the games because their count or clock is invalid.",)

        if not (0 <= arguments.increment_milliseconds <= 60000) or not (2 <= arguments.maximum_plies <= 4096):
            raise ValueError("Failed to prepare the games because their increment or ply limit is invalid.",)

        if arguments.output.exists() or (arguments.candidate.resolve() == arguments.stockfish.resolve()):
            raise ValueError("Failed to prepare the games because the output exists or the binaries are identical.",)

        candidate_options = {"Threads": "1", "Hash": "16",}
        candidate_options.update(parse_options(arguments.candidate_option,),)
        stockfish_options = {"Threads": "1", "Hash": "16",}
        stockfish_options.update(parse_options(arguments.stockfish_option,),)
        openings = []

        for text in arguments.openings.read_text().splitlines():
            if not text.strip():
                continue

            board = chess.Board()
            operations = board.set_epd(text,)

            if not board.is_valid() or board.is_game_over(claim_draw=True,):
                raise ValueError("Failed to prepare a legal, nonterminal opening position.",)

            opening_name = str(operations.get(
                "id",
                "Opening position",
            ),)

            openings.append((board.fen(), opening_name,),)

        if not openings:
            raise ValueError("Failed to prepare the games because the opening suite is empty.",)

        random.Random(arguments.seed,).shuffle(openings,)
        binary_digests = [compute_digest(arguments.candidate,), compute_digest(arguments.stockfish,),]

        candidate = UciClient(
            arguments.candidate,
            candidate_options,
        )

        clients.append(candidate,)

        stockfish = UciClient(
            arguments.stockfish,
            stockfish_options,
        )

        clients.append(stockfish,)

        arguments.output.mkdir(
            parents=True,
            exist_ok=False,
        )

        manifest = {
            "candidate": {
                "path": str(arguments.candidate.resolve(),),
                "sha256": binary_digests[0],
                "identity": candidate.identity,
                "options": candidate_options,
            },
            "stockfish": {
                "path": str(arguments.stockfish.resolve(),),
                "sha256": binary_digests[1],
                "identity": stockfish.identity,
                "options": stockfish_options,
            },
            "openings_sha256": compute_digest(arguments.openings,),
            "seed": arguments.seed,
            "pair_count": arguments.pairs,
            "clock_milliseconds": arguments.clock_milliseconds,
            "increment_milliseconds": arguments.increment_milliseconds,
            "maximum_plies": arguments.maximum_plies,
            "draw_claims": "Claim available rule draws automatically.",
            "platform": platform.platform(),
            "python_chess_version": chess.__version__,
            "started_at_unix_seconds": time.time(),
        }

        for entry in (manifest["candidate"], manifest["stockfish"],):
            network_path = entry["options"].get("EvalFile",)

            if network_path and Path(network_path,).is_file():
                entry["network_sha256"] = compute_digest(Path(network_path,),)

        (arguments.output / "manifest.json").write_text(json.dumps(
            manifest,
            indent=4,
        ) + "\n",)

        wins = 0
        draws = 0
        losses = 0
        pentanomial_counts = [0,] * 5
        games = []

        with (arguments.output / "games.pgn").open("w",) as pgn_output, (arguments.output / "protocol.jsonl").open("w",) as protocol_output:
            for pair_index in range(arguments.pairs,):
                opening_fen, opening_name = openings[pair_index % len(openings,)]
                pair_score = 0.0

                for color_index in range(2,):
                    for client in clients:
                        client.start_game()

                    board = chess.Board(opening_fen,)
                    game = chess.pgn.Game()
                    game.setup(board,)
                    candidate_color = chess.WHITE if color_index == 0 else chess.BLACK
                    game.headers["Event"] = "Alekhine versus Stockfish verification"
                    game.headers["Round"] = f"{pair_index + 1}.{color_index + 1}"
                    game.headers["White"] = candidate.identity if candidate_color == chess.WHITE else stockfish.identity
                    game.headers["Black"] = stockfish.identity if candidate_color == chess.WHITE else candidate.identity
                    game.headers["Opening"] = opening_name
                    game.headers["TimeControl"] = f"{arguments.clock_milliseconds / 1000:g}+{arguments.increment_milliseconds / 1000:g}"
                    remaining_milliseconds = {chess.WHITE: float(arguments.clock_milliseconds,), chess.BLACK: float(arguments.clock_milliseconds,),}
                    moves = []
                    node = game
                    termination = ""
                    winner = None
                    has_forfeit = False

                    while not board.is_game_over(claim_draw=True,) and (len(moves,) < arguments.maximum_plies):
                        moving_color = board.turn
                        client = candidate if moving_color == candidate_color else stockfish
                        position_command = f"position fen {opening_fen}"

                        if moves:
                            position_command += " moves " + " ".join(moves,)

                        white_milliseconds = max(
                            1,
                            int(remaining_milliseconds[chess.WHITE],),
                        )

                        black_milliseconds = max(
                            1,
                            int(remaining_milliseconds[chess.BLACK],),
                        )

                        search_command = f"go wtime {white_milliseconds} btime {black_milliseconds} winc {arguments.increment_milliseconds} binc {arguments.increment_milliseconds}"

                        try:
                            move_name, transcript, elapsed_seconds = client.search_position(
                                position_command,
                                search_command,
                                remaining_milliseconds[moving_color] / 1000 + 0.5,
                            )
                        except (OSError, RuntimeError, TimeoutError,) as error:
                            termination = f"Engine failure: {error}"
                            winner = not moving_color
                            has_forfeit = True

                            break

                        remaining_milliseconds[moving_color] -= elapsed_seconds * 1000

                        protocol_output.write(json.dumps({
                            "pair_index": pair_index,
                            "color_index": color_index,
                            "ply": len(moves,),
                            "fen": board.fen(),
                            "engine": client.identity,
                            "search_command": search_command,
                            "best_move": move_name,
                            "elapsed_milliseconds": elapsed_seconds * 1000,
                            "response": transcript,
                        },) + "\n",)

                        protocol_output.flush()

                        if remaining_milliseconds[moving_color] < 0:
                            termination = "Time forfeit."
                            winner = None if board.has_insufficient_material(not moving_color,) else not moving_color
                            has_forfeit = True

                            break

                        try:
                            chess_move = chess.Move.from_uci(move_name,)
                        except ValueError:
                            chess_move = chess.Move.null()

                        if chess_move not in board.legal_moves:
                            termination = f"Illegal move: {move_name}."
                            winner = not moving_color
                            has_forfeit = True

                            break

                        board.push(chess_move,)
                        moves.append(move_name,)
                        remaining_milliseconds[moving_color] += arguments.increment_milliseconds
                        node = node.add_variation(chess_move,)
                        reports = parse_variations(transcript,)

                        if reports:
                            node.comment = f"Depth {reports[0]['depth_plies']}; elapsed {elapsed_seconds * 1000:.1f} milliseconds."

                    if not has_forfeit:
                        outcome = board.outcome(claim_draw=True,)

                        if outcome is None:
                            termination = "Draw adjudicated at the configured ply limit."
                        else:
                            winner = outcome.winner

                            termination = outcome.termination.name.lower().replace(
                                "_",
                                " ",
                            ).capitalize() + "."

                    score = 0.5 if winner is None else float(winner == candidate_color,)
                    pair_score += score
                    wins += int(score == 1.0,)
                    draws += int(score == 0.5,)
                    losses += int(score == 0.0,)
                    game.headers["Result"] = "1/2-1/2" if winner is None else "1-0" if winner == chess.WHITE else "0-1"
                    game.headers["Termination"] = termination

                    game_record = {
                        "pair_index": pair_index,
                        "candidate_color": "white" if candidate_color == chess.WHITE else "black",
                        "result": game.headers["Result"],
                        "candidate_score": score,
                        "termination": termination,
                        "ply_count": len(moves,),
                        "has_forfeit": has_forfeit,
                    }

                    games.append(game_record,)

                    print(
                        f"Completed game {len(games,)}: candidate score {score:g}; {termination}",
                        flush=True,
                    )

                    pgn_output.write(str(game,) + "\n\n",)
                    pgn_output.flush()

                    summary = {
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "score_fraction": (wins + draws * 0.5) / len(games,),
                        "games": games,
                        "pentanomial_counts": pentanomial_counts,
                        "is_complete": False,
                    }

                    (arguments.output / "summary.json").write_text(json.dumps(
                        summary,
                        indent=4,
                    ) + "\n",)

                pentanomial_counts[int(pair_score * 2,)] += 1

        if binary_digests != [compute_digest(arguments.candidate,), compute_digest(arguments.stockfish,),]:
            raise ValueError("Failed to verify the games because an engine binary changed during the match.",)

        summary["pentanomial_counts"] = pentanomial_counts
        summary["is_complete"] = True
        summary["completed_at_unix_seconds"] = time.time()

        (arguments.output / "summary.json").write_text(json.dumps(
            summary,
            indent=4,
        ) + "\n",)

        print(f"Completed {len(games,)} games: {wins} wins, {draws} draws, and {losses} losses. Results are in {arguments.output}.",)
    except ImportError as error:
        parser.exit(
            1,
            f"Failed to run the games because python-chess is required: {error}\n",
        )
    except (OSError, RuntimeError, ValueError, TimeoutError,) as error:
        parser.exit(
            1,
            f"Failed to run the games: {error}\n",
        )
    finally:
        for client in clients:
            client.close_process()

if __name__ == "__main__":
    run_games()
