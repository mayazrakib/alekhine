import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from uci_client import parse_options

PROJECT_DIRECTORY = Path(__file__,).resolve().parent.parent

def compute_file_digest(path: Path,) -> str:
    with path.open("rb",) as source:
        return hashlib.file_digest(
            source,
            "sha256",
        ).hexdigest()

def build_match_command(arguments: argparse.Namespace,) -> list[str]:
    for path in (arguments.candidate, arguments.baseline, arguments.openings,):
        if not path.is_file():
            raise ValueError(f"Failed to prepare the match because {path} is not a file.",)

    candidate_options = {"Threads": "1", "Hash": str(arguments.hash_mib,),}
    candidate_options.update(parse_options(arguments.candidate_option,),)
    baseline_options = {"Threads": "1", "Hash": str(arguments.hash_mib,),}
    baseline_options.update(parse_options(arguments.baseline_option,),)

    if (arguments.candidate.resolve() == arguments.baseline.resolve()) and (candidate_options == baseline_options):
        raise ValueError("Failed to prepare the match because candidate and baseline must have distinct binaries or options.",)

    for path in (arguments.candidate, arguments.baseline,):
        if not os.access(
            path,
            os.X_OK,
        ):
            raise ValueError(f"Failed to prepare the match because {path} is not executable.",)

    if arguments.output.exists():
        raise ValueError("Failed to prepare the match because the output directory already exists.",)

    if arguments.openings.suffix.lower() not in {".epd", ".pgn",}:
        raise ValueError("Failed to prepare the match because the opening suite must be EPD or PGN.",)

    if not (1 <= arguments.concurrency <= min(
        32,
        os.cpu_count() or 1,
    )):
        raise ValueError("Failed to prepare the match because concurrency exceeds the supported host budget.",)

    if not (1 <= arguments.rounds <= 1000000) or not (1 <= arguments.hash_mib <= 512):
        raise ValueError("Failed to prepare the match because rounds or hash size are outside the supported range.",)

    if not (0 < arguments.elo1 <= 20) or not (0.0 < arguments.alpha < 0.5) or not (0.0 < arguments.beta < 0.5):
        raise ValueError("Failed to prepare the match because its SPRT parameters are invalid.",)

    command = [
        arguments.runner,
        "-engine",
        f"cmd={arguments.candidate.resolve()}",
        "name=candidate",
        *[f"option.{name}={option_value}" for name, option_value in candidate_options.items()],
        "-engine",
        f"cmd={arguments.baseline.resolve()}",
        "name=baseline",
        *[f"option.{name}={option_value}" for name, option_value in baseline_options.items()],
        "-each",
        "proto=uci",
        f"tc={arguments.time_control}",
        "-openings",
        f"file={arguments.openings.resolve()}",
        f"format={arguments.openings.suffix.lower()[1:]}",
        "order=random",
        "-srand",
        str(arguments.seed,),
        "-rounds",
        str(arguments.rounds,),
        "-repeat",
        "-concurrency",
        str(arguments.concurrency,),
        "-sprt",
        "elo0=0",
        f"elo1={arguments.elo1}",
        f"alpha={arguments.alpha}",
        f"beta={arguments.beta}",
        "model=logistic",
        "-report",
        "penta=true",
        "-pgnout",
        f"file={arguments.output.resolve() / 'games.pgn'}",
        "append=false",
        "nodes=true",
        "-log",
        f"file={arguments.output.resolve() / 'runner.log'}",
        "engine=true",
        "append=false",
        "-config",
        f"outname={arguments.output.resolve() / 'resume.json'}",
    ]

    return command

def run_match() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible paired engine matches with Fastchess and SPRT.",)

    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--candidate_option",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--baseline_option",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--openings",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--runner",
        default="fastchess",
    )

    parser.add_argument(
        "--time-control",
        default="10+0.1",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--hash-mib",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261004,
    )

    parser.add_argument(
        "--elo1",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=86400,
    )

    parser.add_argument(
        "--dry-run",
        dest="is_dry_run",
        action="store_true",
    )

    arguments = parser.parse_args()

    try:
        command = build_match_command(arguments,)

        if arguments.is_dry_run:
            print(shlex.join(command,),)

            return

        if shutil.which(arguments.runner,) is None:
            raise ValueError("Failed to run the match because Fastchess is not installed.",)

        if not (1 <= arguments.timeout_seconds <= 604800):
            raise ValueError("Failed to run the match because its timeout must be between 1 second and 7 days.",)

        arguments.output.mkdir(
            parents=True,
            exist_ok=False,
        )

        manifest = {
            "candidate_sha256": compute_file_digest(arguments.candidate,),
            "baseline_sha256": compute_file_digest(arguments.baseline,),
            "openings_sha256": compute_file_digest(arguments.openings,),
            "started_at_unix_seconds": time.time(),
            "command": command,
            "candidate_options": parse_options(arguments.candidate_option,),
            "baseline_options": parse_options(arguments.baseline_option,),
        }

        for name, settings in (("candidate", arguments.candidate_option,), ("baseline", arguments.baseline_option,),):
            network_path = parse_options(settings,).get("EvalFile",)

            if network_path and Path(network_path,).is_file():
                manifest[f"{name}_network_sha256"] = compute_file_digest(Path(network_path,),)

        (arguments.output / "manifest.json").write_text(json.dumps(
            manifest,
            indent=4,
        ) + "\n",)

        subprocess.run(
            command,
            check=True,
            timeout=arguments.timeout_seconds,
        )

        print(f"Completed the match in {arguments.output}.",)
    except (OSError, ValueError, subprocess.SubprocessError,) as error:
        parser.exit(
            1,
            f"Failed to run the match: {error}\n",
        )

if __name__ == "__main__":
    run_match()
