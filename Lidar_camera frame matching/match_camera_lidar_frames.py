
from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from pathlib import Path

from collect_time_files import collect_timed_files

DEFAULT_CAMERA_FOLDER = Path(
    r"D:\Thesis trial\2025-11-03-14-36-01\camera\png"
)

DEFAULT_LIDAR_FOLDER = Path(
    r"D:\Thesis trial\2025-11-03-14-36-01\lidar\pcds"
)

DEFAULT_OUTPUT_FOLDER = Path(
    r"D:\Thesis trial\lidar_and_camera_frame_matching"
)

DEFAULT_MAX_DIFFERENCE_MS = 10.0

# Supported timestamp example:
# 2025-11-03-14-36-01-612
TIMESTAMP_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3,6})"
)

CAMERA_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

LIDAR_EXTENSIONS = {
    ".pcd"
}


@dataclass(frozen=True)
class TimedFile:
    path: Path
    timestamp: datetime
    timestamp_text: str


@dataclass(frozen=True)
class MatchedPair:
    camera: TimedFile
    lidar: TimedFile
    difference_seconds: float


def find_nearest_lidar(
    camera: TimedFile,
    lidar_files: list[TimedFile],
) -> MatchedPair:
    """
    Find the LiDAR frame with the smallest timestamp difference.
    """
    if not lidar_files:
        raise ValueError("No LiDAR files are available.")

    # Binary search manually to avoid external dependencies.
    left = 0
    right = len(lidar_files)

    while left < right:
        middle = (left + right) // 2

        if lidar_files[middle].timestamp < camera.timestamp:
            left = middle + 1
        else:
            right = middle

    candidates: list[TimedFile] = []

    if left < len(lidar_files):
        candidates.append(lidar_files[left])

    if left > 0:
        candidates.append(lidar_files[left - 1])

    nearest = min(
        candidates,
        key=lambda item: abs(
            (item.timestamp - camera.timestamp).total_seconds()
        ),
    )

    difference_seconds = abs(
        (nearest.timestamp - camera.timestamp).total_seconds()
    )

    return MatchedPair(
        camera=camera,
        lidar=nearest,
        difference_seconds=difference_seconds,
    )


def match_frames(
    camera_files: list[TimedFile],
    lidar_files: list[TimedFile],
    maximum_difference_seconds: float,
    allow_reuse: bool,
) -> tuple[list[MatchedPair], list[TimedFile]]:
    """
    Match each camera frame to its nearest LiDAR frame.

    If allow_reuse=False, one LiDAR frame can be assigned only once.
    """
    matched_pairs: list[MatchedPair] = []
    unmatched_cameras: list[TimedFile] = []
    used_lidar_paths: set[Path] = set()

    for camera in camera_files:
        if allow_reuse:
            available_lidar = lidar_files
        else:
            available_lidar = [
                lidar
                for lidar in lidar_files
                if lidar.path not in used_lidar_paths
            ]

        if not available_lidar:
            unmatched_cameras.append(camera)
            continue

        match = find_nearest_lidar(camera, available_lidar)

        if match.difference_seconds > maximum_difference_seconds:
            unmatched_cameras.append(camera)
            continue

        matched_pairs.append(match)
        used_lidar_paths.add(match.lidar.path)

    return matched_pairs, unmatched_cameras


def safe_copy(source: Path, destination: Path) -> None:
    """
    Copy a file while creating its destination directory.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def export_matches(
    matched_pairs: Iterable[MatchedPair],
    unmatched_cameras: Iterable[TimedFile],
    output_folder: Path,
) -> None:
    """
    Create one folder per matched frame pair and write a manifest CSV.

    Output example:
        matched_frames/
            pair_000001/
                camera_000001_2025-11-03-14-36-01-612.png
                lidar_000001_2025-11-03-14-36-01-615.pcd
            pair_000002/
                ...
            matched_frames.csv
            unmatched_camera_frames.csv
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    manifest_path = output_folder / "matched_frames.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "pair_index",
                "camera_original_name",
                "lidar_original_name",
                "camera_timestamp",
                "lidar_timestamp",
                "time_difference_ms",
                "camera_output_path",
                "lidar_output_path",
            ]
        )

        for pair_index, pair in enumerate(matched_pairs, start=1):
            pair_name = f"pair_{pair_index:06d}"
            pair_folder = output_folder / pair_name

            camera_extension = pair.camera.path.suffix.lower()
            lidar_extension = pair.lidar.path.suffix.lower()

            camera_output_name = (
                f"camera_{pair_index:06d}_"
                f"{pair.camera.timestamp_text}"
                f"{camera_extension}"
            )

            lidar_output_name = (
                f"lidar_{pair_index:06d}_"
                f"{pair.lidar.timestamp_text}"
                f"{lidar_extension}"
            )

            camera_output_path = pair_folder / camera_output_name
            lidar_output_path = pair_folder / lidar_output_name

            safe_copy(pair.camera.path, camera_output_path)
            safe_copy(pair.lidar.path, lidar_output_path)

            writer.writerow(
                [
                    pair_index,
                    pair.camera.path.name,
                    pair.lidar.path.name,
                    pair.camera.timestamp_text,
                    pair.lidar.timestamp_text,
                    f"{pair.difference_seconds * 1000:.3f}",
                    camera_output_path.relative_to(output_folder),
                    lidar_output_path.relative_to(output_folder),
                ]
            )

    unmatched_path = output_folder / "unmatched_camera_frames.csv"

    with unmatched_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "camera_original_name",
                "camera_timestamp",
                "camera_original_path",
            ]
        )

        for camera in unmatched_cameras:
            writer.writerow(
                [
                    camera.path.name,
                    camera.timestamp_text,
                    camera.path,
                ]
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match camera frames with the nearest LiDAR PCD frames "
            "using timestamps in their filenames."
        )
    )

    parser.add_argument(
        "--camera-folder",
        type=Path,
        default=DEFAULT_CAMERA_FOLDER,
        help=f"Camera image folder. Default: {DEFAULT_CAMERA_FOLDER}",
    )

    parser.add_argument(
        "--lidar-folder",
        type=Path,
        default=DEFAULT_LIDAR_FOLDER,
        help=f"LiDAR PCD folder. Default: {DEFAULT_LIDAR_FOLDER}",
    )

    parser.add_argument(
        "--output-folder",
        type=Path,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder. Default: {DEFAULT_OUTPUT_FOLDER}",
    )

    parser.add_argument(
        "--max-difference-ms",
        type=float,
        default=DEFAULT_MAX_DIFFERENCE_MS,
        help=(
            "Maximum permitted camera–LiDAR timestamp difference "
            f"in milliseconds. Default: {DEFAULT_MAX_DIFFERENCE_MS}"
        ),
    )

    parser.add_argument(
        "--allow-lidar-reuse",
        action="store_true",
        help=(
            "Allow the same LiDAR frame to be matched with multiple "
            "camera frames. Normally unnecessary when LiDAR has a "
            "higher frame rate."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    camera_folder = args.camera_folder
    lidar_folder = args.lidar_folder
    output_folder = args.output_folder
    max_difference_ms = args.max_difference_ms

    if args.max_difference_ms < 0:
        raise ValueError("--max-difference-ms must be non-negative.")

    print("Reading camera frames...")
    camera_files = collect_timed_files(
        args.camera_folder,
        CAMERA_EXTENSIONS,
    )

    print("Reading LiDAR frames...")
    lidar_files = collect_timed_files(
        args.lidar_folder,
        LIDAR_EXTENSIONS,
    )

    if not camera_files:
        raise RuntimeError(
            "No timestamped camera images were found."
        )

    if not lidar_files:
        raise RuntimeError(
            "No timestamped LiDAR PCD files were found."
        )

    print(f"\nCamera frames found: {len(camera_files)}")
    print(f"LiDAR frames found:  {len(lidar_files)}")

    matched_pairs, unmatched_cameras = match_frames(
        camera_files=camera_files,
        lidar_files=lidar_files,
        maximum_difference_seconds=(
            args.max_difference_ms / 1000.0
        ),
        allow_reuse=args.allow_lidar_reuse,
    )

    export_matches(
        matched_pairs=matched_pairs,
        unmatched_cameras=unmatched_cameras,
        output_folder=args.output_folder,
    )

    print("\nMatching completed.")
    print(f"Matched pairs:          {len(matched_pairs)}")
    print(f"Unmatched camera frames: {len(unmatched_cameras)}")
    print(f"Output folder:          {args.output_folder.resolve()}")

    if matched_pairs:
        differences_ms = [
            pair.difference_seconds * 1000
            for pair in matched_pairs
        ]

        print(
            f"Minimum time difference: {min(differences_ms):.3f} ms"
        )
        print(
            f"Average time difference: "
            f"{sum(differences_ms) / len(differences_ms):.3f} ms"
        )
        print(
            f"Maximum time difference: {max(differences_ms):.3f} ms"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        raise
