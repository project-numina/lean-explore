"""Script to prepare and release data files for a specific toolchain version.
This script calculates SHA256 checksums for key data files and updates the manifest.json
file with the new version information and checksums.
"""  # noqa: D205

import argparse
import gzip
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_MANIFEST_FILE = _PROJECT_ROOT / "manifest.json"
_DATA_DIR = _PROJECT_ROOT / "data"
_RELEASES_DIR = _DATA_DIR / "releases"

_DATABASE_FILE = _DATA_DIR / "lean_explore_data.db"
_FAISS_INDEX_FILE = _DATA_DIR / "main_faiss.index"
_FAISS_MAP_FILE = _DATA_DIR / "faiss_ids_map.json"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def calculate_sha256_checksum(file_path: Path) -> str:
    """Calculate the SHA256 checksum of a file.

    Args:
        file_path (Path): Path to the file.

    Returns:
        str: SHA256 checksum as a hexadecimal string.
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compress_file_(file_path: Path, out_floder: Path) -> Path:
    """Compress a file using gzip.

    Args:
        file_path (Path): Path to the file to compress.
        out_floder (Path): Directory where the compressed file will be saved.
    """
    compressed_file_path = (
        out_floder / file_path.with_suffix(file_path.suffix + ".gz").name
    )
    with file_path.open("rb") as f_in:
        with gzip.open(compressed_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    logger.info(f"Compressed {file_path.name} to {compressed_file_path}")
    return compressed_file_path


def release(version: str, description: str):
    """Prepare and release data files for a specific toolchain version.

    Args:
        version (str): The toolchain version to release (e.g., "0.1.0").
        description (str): Optional description for the release.
    """
    logger.info(f"Releasing data for toolchain version: {version}")

    # Ensure the data files exists
    for file_path in [_DATABASE_FILE, _FAISS_INDEX_FILE, _FAISS_MAP_FILE]:
        if not file_path.exists():
            logger.error(f"Required file not found: {file_path}")
            sys.exit(1)

    # Ensure the releases directory exists
    version_dir = _RELEASES_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Load existing manifest
    if not _MANIFEST_FILE.exists():
        logger.error(f"Manifest file not found: {_MANIFEST_FILE}")
        sys.exit(1)

    manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))

    # Ensure the version is not already present
    if version in manifest.get("toolchains", {}):
        logger.error(f"Version {version} already exists in the manifest.")
        sys.exit(1)

    # Define the files to process
    files_to_process = {
        "database": _DATABASE_FILE,
        "faiss_index": _FAISS_INDEX_FILE,
        "faiss_map": _FAISS_MAP_FILE,
    }

    checksums = {}
    size_bytes_compressed = {}
    size_bytes_uncompressed = {}

    # Process each file: compress and calculate checksum
    # We compress the files to save space and ensure integrity
    for name, file_path in files_to_process.items():
        compressed_file_path = compress_file_(file_path, version_dir)
        size_bytes_uncompressed[name] = file_path.stat().st_size
        size_bytes_compressed[name] = compressed_file_path.stat().st_size
        logger.info(
            f"File {file_path.name}: "
            f"uncompressed size = {size_bytes_uncompressed[name]} bytes, "
            f"compressed size = {size_bytes_compressed[name]} bytes"
        )

        checksum = calculate_sha256_checksum(compressed_file_path)
        checksums[name] = checksum
        logger.info(
            f"Calculated SHA256 for compressed {compressed_file_path.name}: {checksum}"
        )

    files = [
        {
            "local_name": file_path.name,
            "remote_name": file_path.name + ".gz",
            "sha256": checksums[name],
            "size_bytes_compressed": size_bytes_compressed[name],
            "size_bytes_uncompressed": size_bytes_uncompressed[name],
        }
        for name, file_path in files_to_process.items()
    ]

    manifest["latest_manifest_version"] = version
    manifest["toolchains"][version] = {
        "description": description or f"Release for version {version}",
        "release_date": datetime.now().strftime("%Y-%m-%d"),
        "assets_base_path_r2": f"releases/{version}/",
        "files": files,
    }

    # Write updated manifest back to file
    _MANIFEST_FILE.write_text(json.dumps(manifest, indent=4), encoding="utf-8")

    logger.info(f"Updated manifest file at {_MANIFEST_FILE}")
    logger.info("Release process completed successfully.")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare and release data files for a specific toolchain version.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Toolchain version to process (e.g., '0.1.0').",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional description for the release.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    release(args.version, args.description)
