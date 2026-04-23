#!/usr/bin/env python3
"""Generate a simple visual occupancy board from model predictions."""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import sys
import tarfile
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parking_project.model import SimpleParkingCNN

LABEL_NAMES = {0: "Empty", 1: "Occupied"}
LABEL_COLORS = {0: (28, 166, 87), 1: (214, 68, 68)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained model weights")
    parser.add_argument("--output-image", type=Path, required=True, help="Destination PNG path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=24, help="Number of parking spots to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
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


def load_rows(csv_path: Path, limit: int | None, seed: int, num_samples: int) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:limit]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:num_samples]


def open_image(row: dict[str, str], archive_path: Path, extracted_root: Path | None) -> Image.Image:
    if extracted_root is not None:
        return Image.open(extracted_root / row["archive_member"]).convert("RGB")

    with tarfile.open(archive_path, "r") as archive:
        member_handle = archive.extractfile(row["archive_member"])
        if member_handle is None:
            raise FileNotFoundError(f"Could not read {row['archive_member']}")
        return Image.open(io.BytesIO(member_handle.read())).convert("RGB")


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.resize((image_size, image_size))
    array = torch.tensor(list(image.getdata()), dtype=torch.float32).view(image_size, image_size, 3)
    array = array.permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=array.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=array.dtype).view(3, 1, 1)
    return (array - mean) / std


@torch.no_grad()
def predict_rows(
    rows: list[dict[str, str]],
    model: SimpleParkingCNN,
    archive_path: Path,
    extracted_root: Path | None,
    image_size: int,
    device: torch.device,
) -> list[dict]:
    results: list[dict] = []
    model.eval()
    for row in rows:
        image = open_image(row, archive_path, extracted_root)
        tensor = preprocess_image(image, image_size).unsqueeze(0).to(device)
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu()[0]
        prediction = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[prediction].item())
        results.append(
            {
                "row": row,
                "image": image,
                "prediction": prediction,
                "confidence": confidence,
                "target": int(row["label_id"]),
            }
        )
    return results


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_board(predictions: list[dict], destination: Path, title: str) -> None:
    cols = 4
    rows = math.ceil(len(predictions) / cols)
    card_w, card_h = 220, 170
    gap = 18
    margin = 32
    header_h = 150
    width = margin * 2 + cols * card_w + (cols - 1) * gap
    height = margin * 2 + header_h + rows * card_h + (rows - 1) * gap

    board = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(board)

    title_font = font(30)
    body_font = font(18)
    small_font = font(15)

    total = len(predictions)
    occupied = sum(1 for item in predictions if item["prediction"] == 1)
    empty = total - occupied
    occupancy_rate = occupied / total if total else 0.0

    draw.text((margin, margin), title, fill=(26, 43, 60), font=title_font)
    draw.text(
        (margin, margin + 46),
        f"Predicted occupied: {occupied} / {total} ({occupancy_rate:.1%})",
        fill=(55, 73, 92),
        font=body_font,
    )
    draw.text(
        (margin, margin + 76),
        f"Predicted empty: {empty} / {total}",
        fill=(55, 73, 92),
        font=body_font,
    )

    legend_y = margin + 112
    draw.rounded_rectangle((margin, legend_y, margin + 18, legend_y + 18), radius=4, fill=LABEL_COLORS[0])
    draw.text((margin + 28, legend_y - 2), "Empty", fill=(55, 73, 92), font=small_font)
    draw.rounded_rectangle((margin + 110, legend_y, margin + 128, legend_y + 18), radius=4, fill=LABEL_COLORS[1])
    draw.text((margin + 138, legend_y - 2), "Occupied", fill=(55, 73, 92), font=small_font)

    start_y = margin + header_h
    thumb_w, thumb_h = 190, 96
    for idx, item in enumerate(predictions):
        row_idx, col_idx = divmod(idx, cols)
        x = margin + col_idx * (card_w + gap)
        y = start_y + row_idx * (card_h + gap)
        color = LABEL_COLORS[item["prediction"]]

        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=16, fill=(255, 255, 255), outline=color, width=4)

        thumb = item["image"].resize((thumb_w, thumb_h))
        board.paste(thumb, (x + 15, y + 14))

        label = LABEL_NAMES[item["prediction"]]
        confidence = item["confidence"]
        truth = LABEL_NAMES[item["target"]]
        correct = "correct" if item["prediction"] == item["target"] else "wrong"

        draw.text((x + 15, y + 116), f"Prediction: {label}", fill=color, font=body_font)
        draw.text((x + 15, y + 138), f"Confidence: {confidence:.1%} | Truth: {truth} | {correct}", fill=(75, 90, 105), font=small_font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    board.save(destination)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = select_device(config["training"]["device"])

    model = SimpleParkingCNN(num_classes=config["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    data_cfg = config["data"]
    rows = load_rows(
        csv_path=PROJECT_ROOT / data_cfg["metadata_dir"] / f"{args.split}.csv",
        limit=data_cfg.get(f"max_{args.split}_samples"),
        seed=args.seed,
        num_samples=args.num_samples,
    )
    predictions = predict_rows(
        rows=rows,
        model=model,
        archive_path=PROJECT_ROOT / data_cfg["archive_path"],
        extracted_root=PROJECT_ROOT / data_cfg["extracted_root"] if data_cfg.get("extracted_root") else None,
        image_size=data_cfg["image_size"],
        device=device,
    )
    draw_board(predictions, args.output_image, title="Smart Parking Occupancy Board")
    print(f"Saved system output demo to {args.output_image}")


if __name__ == "__main__":
    main()
