import shutil
import subprocess
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__,).resolve().parent.parent
TRAINING_RECORDS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 100 | 1 | e2e4:0.7 d2d4:0.3 | game_0\n"
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 | 100 | 1 | e7e5:1 | game_0\n"
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1 | 100 | 1 | d7d5:1 | game_3\n"
)

def create_uci_fixture(directory: Path,) -> Path:
    fixture_path = directory / "protocol_fixture"
    shutil.copyfile(
        PROJECT_DIRECTORY / "test" / "uci_fixture.py",
        fixture_path,
    )
    fixture_path.chmod(0o700,)

    return fixture_path

def train_fixture_network(
    executable: Path,
    directory: Path,
) -> Path:
    dataset_path = directory / "positions.txt"
    network_path = directory / "model.nnue"
    dataset_path.write_text(TRAINING_RECORDS,)
    subprocess.run(
        [str(executable,), "train_network", str(dataset_path,), str(network_path,), "2", "--batch_size", "2", "--seed", "42",],
        capture_output=True,
        text=True,
        check=True,
        timeout=30.0,
    )

    return network_path
