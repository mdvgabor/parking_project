#!/usr/bin/env python3
"""Evaluate a trained model overall and across metadata groups."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parking_project.data import PKLotTarDataset
from parking_project.model import SimpleParkingCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to saved model weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for robustness outputs",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset(config: dict, split_name: str) -> PKLotTarDataset:
    data_cfg = config["data"]
    return PKLotTarDataset(
        archive_path=PROJECT_ROOT / data_cfg["archive_path"],
        metadata_csv=PROJECT_ROOT / data_cfg["metadata_dir"] / f"{split_name}.csv",
        extracted_root=PROJECT_ROOT / data_cfg["extracted_root"] if data_cfg.get("extracted_root") else None,
        image_size=data_cfg["image_size"],
        limit=data_cfg.get(f"max_{split_name}_samples"),
        seed=config["seed"],
        train=False,
        augmentation={},
        return_metadata=True,
    )


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[list[dict[str, str]], list[int], list[int], float]:
    rows: list[dict[str, str]] = []
    predictions: list[int] = []
    targets: list[int] = []
    running_loss = 0.0

    model.eval()
    for inputs, labels, metadata in dataloader:
        inputs = inputs.to(device)
        labels_device = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels_device)
        running_loss += loss.item() * inputs.size(0)

        preds = torch.argmax(logits, dim=1).cpu().tolist()
        predictions.extend(preds)
        targets.extend(labels.tolist())

        batch_size = len(preds)
        for i in range(batch_size):
            row = {key: metadata[key][i] for key in metadata}
            rows.append(row)

    average_loss = running_loss / len(dataloader.dataset)
    return rows, targets, predictions, average_loss


def compute_metrics(targets: list[int], predictions: list[int], loss: float | None = None) -> dict[str, float]:
    result = {
        "count": len(targets),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
    }
    if loss is not None:
        result["loss"] = loss
    return result


def group_metrics(rows: list[dict[str, str]], targets: list[int], predictions: list[int], field: str) -> list[dict[str, float | str]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped_indices[row[field]].append(idx)

    results: list[dict[str, float | str]] = []
    for group_name, indices in sorted(grouped_indices.items()):
        group_targets = [targets[i] for i in indices]
        group_predictions = [predictions[i] for i in indices]
        metrics = compute_metrics(group_targets, group_predictions)
        metrics[field] = group_name
        results.append(metrics)
    return results


def write_group_csv(rows: list[dict[str, float | str]], destination: Path, group_field: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [group_field, "count", "accuracy", "precision", "recall", "f1"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions_csv(
    rows: list[dict[str, str]],
    targets: list[int],
    predictions: list[int],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "archive_member",
                "site",
                "weather",
                "date",
                "label_id",
                "prediction",
                "correct",
            ]
        )
        for row, target, prediction in zip(rows, targets, predictions):
            writer.writerow(
                [
                    row["archive_member"],
                    row["site"],
                    row["weather"],
                    row["date"],
                    target,
                    prediction,
                    int(target == prediction),
                ]
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = select_device(config["training"]["device"])

    dataset = make_dataset(config, args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    model = SimpleParkingCNN(num_classes=config["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    rows, targets, predictions, loss = collect_predictions(model, dataloader, criterion, device)

    overall = compute_metrics(targets, predictions, loss=loss)
    by_site = group_metrics(rows, targets, predictions, "site")
    by_weather = group_metrics(rows, targets, predictions, "weather")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": args.split,
                "overall": overall,
                "by_site": by_site,
                "by_weather": by_weather,
            },
            handle,
            indent=2,
        )

    write_group_csv(by_site, args.output_dir / "by_site.csv", "site")
    write_group_csv(by_weather, args.output_dir / "by_weather.csv", "weather")
    write_predictions_csv(rows, targets, predictions, args.output_dir / "predictions.csv")

    print(json.dumps({"split": args.split, "overall": overall}, indent=2))
    print(f"Saved robustness outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
