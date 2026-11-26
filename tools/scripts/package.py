import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__,).resolve().parents[2]

def package_project(arguments: argparse.Namespace,) -> None:
    build_directory = PROJECT_DIRECTORY / "build" / arguments.mode
    manifest_path = build_directory / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(),)
    engine_path = build_directory / "alekhine"

    if not engine_path.is_file():
        engine_path = engine_path.with_suffix(".exe",)

    with engine_path.open("rb",) as engine_file:
        binary_digest = hashlib.file_digest(
            engine_file,
            "sha256",
        ).hexdigest()

    if binary_digest != manifest["binary_sha256"]:
        raise ValueError("Failed to package the engine because its binary differs from the build manifest.",)

    archive_name = f"alekhine-{manifest['version']}-{manifest['target_architecture']}-{arguments.mode}"
    output_directory = arguments.output

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_paths = (
        output_directory / f"{archive_name}.tar.gz",
        output_directory / f"{archive_name}.zip",
    )

    if any((path.exists() for path in archive_paths),):
        raise ValueError("Failed to package the engine because an output archive already exists.",)

    members = [
        (engine_path, f"{archive_name}/bin/{engine_path.name}",),
        (manifest_path, f"{archive_name}/build_manifest.json",),
    ]

    for filename in ("LICENSE", "README.md", "project.toml", "Cargo.toml",):
        members.append((PROJECT_DIRECTORY / filename, f"{archive_name}/{filename}",),)

    with tarfile.open(
        archive_paths[0],
        "x:gz",
    ) as archive:
        for source_path, archive_path in members:
            archive.add(
                source_path,
                arcname=archive_path,
            )

    with zipfile.ZipFile(
        archive_paths[1],
        "x",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for source_path, archive_path in members:
            archive.write(
                source_path,
                arcname=archive_path,
            )

    print(f"Packaged alekhine in {output_directory}.",)

def run_package() -> None:
    parser = argparse.ArgumentParser(description="Package a verified alekhine Rust build as executable distribution archives.",)

    parser.add_argument(
        "--mode",
        choices=("release", "debug",),
        default="release",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIRECTORY / "dist",
    )

    arguments = parser.parse_args()

    try:
        package_project(arguments,)
    except (KeyError, OSError, ValueError,) as error:
        parser.exit(
            1,
            f"Failed to package alekhine: {error}\n",
        )

if __name__ == "__main__":
    run_package()
