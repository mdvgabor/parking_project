#!/usr/bin/env python3
"""Train and evaluate a baseline parking occupancy classifier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
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
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "baseline",
        help="Where to save model weights and metrics",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataloader(config: dict, split_name: str, shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
    limit = data_cfg.get(f"max_{split_name}_samples")
    dataset = PKLotTarDataset(
        archive_path=PROJECT_ROOT / data_cfg["archive_path"],
        metadata_csv=PROJECT_ROOT / data_cfg["metadata_dir"] / f"{split_name}.csv",
        extracted_root=PROJECT_ROOT / data_cfg["extracted_root"] if data_cfg.get("extracted_root") else None,
        image_size=data_cfg["image_size"],
        limit=limit,
        seed=config["seed"],
        train=split_name == "train",
        augmentation=data_cfg.get("augmentation", {}),
    )
    return DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=data_cfg["num_workers"],
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    targets_all: list[int] = []
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets_device = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets_device)
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        predictions.extend(preds)
        targets_all.extend(targets.tolist())

    return {
        "loss": running_loss / len(dataloader.dataset),
        "accuracy": accuracy_score(targets_all, predictions),
        "precision": precision_score(targets_all, predictions, zero_division=0),
        "recall": recall_score(targets_all, predictions, zero_division=0),
        "f1": f1_score(targets_all, predictions, zero_division=0),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = select_device(config["training"]["device"])

    train_loader = make_dataloader(config, "train", shuffle=True)
    val_loader = make_dataloader(config, "val", shuffle=False)
    test_loader = make_dataloader(config, "test", shuffle=False)

    model = SimpleParkingCNN(num_classes=config["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    best_epoch = 0
    patience = config["training"].get("patience", 3)
    epochs_without_improvement = 0
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_result = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(epoch_result)
        print(json.dumps(epoch_result))

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_val_f1": best_val_f1,
                        }
                    )
                )
                break

    best_model_path = args.output_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = evaluate(model, test_loader, criterion, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "test": test_metrics,
            },
            handle,
            indent=2,
        )

    print("test_metrics", json.dumps(test_metrics))
    print(f"Saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
