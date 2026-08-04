from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import re

@dataclass(frozen=True)
class TimedFile:
    path: Path
    timestamp: datetime
    timestamp_text: str

TIMESTAMP_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3,6})"
)

def extract_timestamp(file_path: Path) -> Optional[TimedFile]:
    """
    Extract a timestamp from a filename.

    Supported examples:
        camera__2025-11-03-14-36-01-612.png
        lidar_01__2025-11-03-14-36-01-615.pcd

    The last timestamp field may contain 3 digits (milliseconds)
    or up to 6 digits (microseconds).
    """
    match = TIMESTAMP_PATTERN.search(file_path.stem)

    if match is None:
        return None

    timestamp_text = match.group("timestamp")
    parts = timestamp_text.split("-")

    if len(parts) != 7:
        return None

    date_time_text = "-".join(parts[:6])
    fractional_text = parts[6]

    # Convert milliseconds or shorter fractions to six-digit microseconds.
    microseconds = fractional_text.ljust(6, "0")[:6]

    try:
        timestamp = datetime.strptime(
            f"{date_time_text}-{microseconds}",
            "%Y-%m-%d-%H-%M-%S-%f",
        )
    except ValueError:
        return None

    return TimedFile(
        path=file_path,
        timestamp=timestamp,
        timestamp_text=timestamp_text,
    )

def collect_timed_files(
    folder: Path,
    allowed_extensions: set[str],
) -> list[TimedFile]:
    """
    Recursively find supported files and extract timestamps.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    timed_files: list[TimedFile] = []
    skipped_files: list[Path] = []

    for file_path in sorted(folder.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in allowed_extensions:
            continue

        timed_file = extract_timestamp(file_path)

        if timed_file is None:
            skipped_files.append(file_path)
            continue

        timed_files.append(timed_file)

    timed_files.sort(key=lambda item: item.timestamp)

    if skipped_files:
        print("\nFiles skipped because no timestamp was found:")
        for skipped in skipped_files:
            print(f"  - {skipped.name}")

    return timed_files