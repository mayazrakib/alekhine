import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import (
    Literal,
    TypedDict,
)

MAXIMUM_LINE_BYTES = 65536

EngineOptions = dict[
    str,
    str,
]

SearchResponse = tuple[
    str,
    list[str],
    float,
]

class PrincipalVariation(TypedDict,):
    depth_plies: int
    variation_index: int

    score_kind: Literal[
        "cp",
        "mate",
    ]

    score: int
    moves: list[str]

class UciClient:
    def __init__(
        self,
        executable: Path,
        options: EngineOptions,
    ) -> None:
        self.process = subprocess.Popen(
            [str(executable.resolve(),),],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self.lines: queue.Queue[str | None] = queue.Queue(maxsize=1024,)
        self.diagnostics: deque[str] = deque(maxlen=128,)
        self.should_close = threading.Event()

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
        self.identity = ""
        self.advertised_options: EngineOptions = {}
        self.is_searching = False

        try:
            self.send_command("uci",)

            handshake = self.wait_for_line(
                "uciok",
                20.0,
            )

            for line in handshake:
                if line.startswith("id name ",):
                    self.identity = line[8:]
                elif line.startswith("option name ",) and " type " in line:
                    name = line[12:].split(" type ",)[0]
                    self.advertised_options[name] = line

            for name, option_value in options.items():
                if name not in self.advertised_options:
                    raise ValueError(f"Failed to configure {self.identity} because option {name} is unsupported.",)

                self.send_command(f"setoption name {name} value {option_value}",)

            self.send_command("isready",)

            readiness = self.wait_for_line(
                "readyok",
                20.0,
            )

            if any((line.startswith("info string Failed ",) for line in readiness),):
                raise ValueError(f"Failed to configure {self.identity}: {readiness}",)
        except BaseException:
            self.close_process()

            raise

    def enqueue_line(
        self,
        line: str | None,
    ) -> None:
        while not self.should_close.is_set():
            try:
                self.lines.put(
                    line,
                    timeout=0.1,
                )

                return
            except queue.Full:
                continue

    def read_output(self,) -> None:
        try:
            while not self.should_close.is_set():
                line = self.process.stdout.readline(MAXIMUM_LINE_BYTES + 1,)

                if not line:
                    break

                if len(line.encode("utf-8",),) > MAXIMUM_LINE_BYTES:
                    self.diagnostics.append("Rejected oversized engine output.",)

                    break

                self.enqueue_line(line.strip(),)
        except (OSError, ValueError, UnicodeError,) as error:
            self.diagnostics.append(f"Failed to read engine output: {error}",)
        finally:
            self.enqueue_line(None,)

    def read_errors(self,) -> None:
        try:
            while not self.should_close.is_set():
                line = self.process.stderr.readline(MAXIMUM_LINE_BYTES + 1,)

                if not line:
                    break

                self.diagnostics.append(line.strip()[:4096],)
        except (OSError, ValueError, UnicodeError,) as error:
            self.diagnostics.append(f"Failed to read engine diagnostics: {error}",)

    def send_command(
        self,
        command: str,
    ) -> None:
        if ("\n" in command) or ("\r" in command) or (len(command.encode("utf-8",),) > MAXIMUM_LINE_BYTES):
            raise ValueError("Failed to send an invalid UCI command.",)

        if self.process.poll() is not None:
            raise RuntimeError(f"The engine terminated with status {self.process.returncode}: {list(self.diagnostics,)}",)

        self.process.stdin.write(command + "\n",)
        self.process.stdin.flush()

    def wait_for_line(
        self,
        prefix: str,
        timeout_seconds: float,
    ) -> list[str]:
        deadline_seconds = time.monotonic() + timeout_seconds
        transcript = []

        while True:
            remaining_seconds = deadline_seconds - time.monotonic()

            if remaining_seconds <= 0:
                raise TimeoutError(f"Timed out waiting for {prefix} from {self.identity}.",)

            try:
                line = self.lines.get(timeout=remaining_seconds,)
            except queue.Empty as error:
                raise TimeoutError(f"Timed out waiting for {prefix} from {self.identity}.",) from error

            if line is None:
                raise RuntimeError(f"The engine closed its output: {list(self.diagnostics,)}",)

            transcript.append(line,)

            if line.startswith(prefix,):
                return transcript

            if len(transcript,) > 100000:
                raise RuntimeError("Failed to consume an engine response because its line budget was exceeded.",)

    def start_game(self,) -> None:
        if self.is_searching:
            self.send_command("stop",)

            self.wait_for_line(
                "bestmove ",
                2.0,
            )

            self.is_searching = False

        self.send_command("ucinewgame",)
        self.send_command("isready",)

        self.wait_for_line(
            "readyok",
            20.0,
        )

    def search_position(
        self,
        position_command: str,
        search_command: str,
        timeout_seconds: float,
    ) -> SearchResponse:
        self.send_command(position_command,)
        started_at_seconds = time.monotonic()
        self.is_searching = True
        self.send_command(search_command,)

        transcript = self.wait_for_line(
            "bestmove ",
            timeout_seconds,
        )

        elapsed_seconds = time.monotonic() - started_at_seconds
        self.is_searching = False
        bestmove_tokens = transcript[-1].split()

        if len(bestmove_tokens,) < 2:
            raise RuntimeError("Failed to parse the engine's best move.",)

        return bestmove_tokens[1], transcript, elapsed_seconds

    def close_process(self,) -> None:
        if self.process.poll() is None:
            try:
                self.send_command("quit",)
            except (OSError, RuntimeError,) as error:
                self.diagnostics.append(f"Failed to send quit: {error}",)

            try:
                self.process.wait(timeout=2.0,)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

        self.should_close.set()
        self.output_thread.join(timeout=2.0,)
        self.error_thread.join(timeout=2.0,)
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.stderr.close()

def parse_options(settings: list[str],) -> EngineOptions:
    options = {}

    for setting in settings:
        if "=" not in setting:
            raise ValueError("Failed to parse an engine option; use name=value.",)

        name, option_value = setting.split(
            "=",
            1,
        )

        if not name or ("\n" in setting) or ("\r" in setting) or (name in options):
            raise ValueError("Failed to parse an empty, duplicate, or multiline engine option.",)

        options[name] = option_value

    return options

def parse_variations(transcript: list[str],) -> list[PrincipalVariation]:
    depths: dict[
        int,
        dict[
            int,
            PrincipalVariation,
        ],
    ] = {}

    for line in transcript:
        tokens = line.split()

        if not tokens or (tokens[0] != "info") or ("score" not in tokens) or ("pv" not in tokens):
            continue

        if (len(tokens,) > 1) and (tokens[1] == "string"):
            continue

        if ("upperbound" in tokens) or ("lowerbound" in tokens):
            continue

        try:
            depth = int(tokens[tokens.index("depth",) + 1],)
            variation_index = int(tokens[tokens.index("multipv",) + 1],) if "multipv" in tokens else 1
            score_index = tokens.index("score",)
            score_kind = tokens[score_index + 1]
            score = int(tokens[score_index + 2],)
            moves = tokens[tokens.index("pv",) + 1:]

            if not moves or (score_kind not in {"cp", "mate",}) or (depth < 0) or (variation_index < 1):
                continue

            variation = {
                "depth_plies": depth,
                "variation_index": variation_index,
                "score_kind": score_kind,
                "score": score,
                "moves": moves,
            }

            depths.setdefault(
                depth,
                {},
            )[variation_index] = variation
        except (IndexError, ValueError,):
            continue

    for depth in sorted(
        depths,
        reverse=True,
    ):
        if 1 in depths[depth]:
            return [depths[depth][index] for index in sorted(depths[depth],)]

    return []
