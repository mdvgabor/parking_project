#!/usr/bin/env python3
"""Create reproducible PKLot train/validation/test metadata splits."""

from __future__ import annotations

import argparse
import csv
import random
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


LABEL_TO_ID = {"Empty": 0, "Occupied": 1}


@dataclass(frozen=True)
class Sample:
    archive_member: str
    label_name: str
    label_id: int
    site: str
    weather: str
    date: str
    filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Path to PKLot.tar")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated CSV metadata files",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"({train_ratio}, {val_ratio}, {test_ratio})"
        )


def iter_segmented_samples(archive_path: Path) -> list[Sample]:
    samples: list[Sample] = []

    with tarfile.open(archive_path, "r") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".jpg"):
                continue

            parts = member.name.split("/")
            if len(parts) != 7 or parts[1] != "PKLotSegmented":
                continue

            _, _, site, weather, date, label_name, filename = parts
            if label_name not in LABEL_TO_ID:
                continue

            samples.append(
                Sample(
                    archive_member=member.name,
                    label_name=label_name,
                    label_id=LABEL_TO_ID[label_name],
                    site=site,
                    weather=weather,
                    date=date,
                    filename=filename,
                )
            )

    if not samples:
        raise RuntimeError("No segmented PKLot samples found in archive.")

    return samples


def split_group(samples: list[Sample], rng: random.Random, train_ratio: float, val_ratio: float) -> dict[str, list[Sample]]:
    rng.shuffle(samples)
    total = len(samples)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Keep at least one sample in later splits when a group is reasonably sized.
    if total >= 3:
        train_end = min(max(train_end, 1), total - 2)
        val_end = min(max(val_end, train_end + 1), total - 1)

    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }


def create_splits(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    grouped: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.site, sample.label_name)].append(sample)

    rng = random.Random(seed)
    split_samples: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}

    for group_items in grouped.values():
        group_split = split_group(group_items, rng, train_ratio, val_ratio)
        for split_name, split_items in group_split.items():
            split_samples[split_name].extend(split_items)

    for split_name in split_samples:
        split_samples[split_name].sort(key=lambda item: item.archive_member)

    return split_samples


def write_csv(samples: list[Sample], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "archive_member",
                "label_name",
                "label_id",
                "site",
                "weather",
                "date",
                "filename",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.archive_member,
                    sample.label_name,
                    sample.label_id,
                    sample.site,
                    sample.weather,
                    sample.date,
                    sample.filename,
                ]
            )


def write_summary(split_samples: dict[str, list[Sample]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "count", "empty", "occupied"])
        for split_name, samples in split_samples.items():
            label_counts = Counter(sample.label_name for sample in samples)
            writer.writerow(
                [
                    split_name,
                    len(samples),
                    label_counts.get("Empty", 0),
                    label_counts.get("Occupied", 0),
                ]
            )


def main() -> None:
    args = parse_args()
    validate_split_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    samples = iter_segmented_samples(args.archive)
    split_samples = create_splits(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    for split_name, items in split_samples.items():
        write_csv(items, args.output_dir / f"{split_name}.csv")

    write_summary(split_samples, args.output_dir / "summary.csv")

    print(f"Indexed {len(samples)} segmented samples from {args.archive}")
    for split_name, items in split_samples.items():
        label_counts = Counter(sample.label_name for sample in items)
        print(
            f"{split_name}: total={len(items)} "
            f"empty={label_counts.get('Empty', 0)} "
            f"occupied={label_counts.get('Occupied', 0)}"
        )
    print(f"Metadata written to {args.output_dir}")


if __name__ == "__main__":
    main()
