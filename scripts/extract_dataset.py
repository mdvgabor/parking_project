#!/usr/bin/env python3
"""Extract PKLot images referenced by metadata CSV files to regular files."""

from __future__ import annotations

import argparse
import csv
import random
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Path to PKLot.tar")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        required=True,
        help="Directory containing split CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for extracted images",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Metadata split names to extract",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional cap on number of extracted samples per split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for randomized limited extraction")
    return parser.parse_args()


def load_members(csv_path: Path, limit: int | None, seed: int) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        members = [row["archive_member"] for row in reader]
    if limit is not None:
        rng = random.Random(seed)
        rng.shuffle(members)
        members = members[:limit]
    return members


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with tarfile.open(args.archive, "r") as archive:
        for split_name in args.splits:
            csv_path = args.metadata_dir / f"{split_name}.csv"
            members = load_members(csv_path, args.limit_per_split, args.seed)
            written = 0
            for member_name in members:
                destination = args.output_dir / member_name
                if destination.exists():
                    written += 1
                    continue

                destination.parent.mkdir(parents=True, exist_ok=True)
                member = archive.getmember(member_name)
                source = archive.extractfile(member)
                if source is None:
                    raise FileNotFoundError(f"Could not extract {member_name}")
                with destination.open("wb") as handle:
                    handle.write(source.read())
                written += 1

            total_written += written
            print(f"{split_name}: prepared {written} files")

    print(f"Prepared {total_written} files under {args.output_dir}")


if __name__ == "__main__":
    main()
