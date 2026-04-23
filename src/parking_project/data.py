"""Dataset helpers for reading PKLot samples from metadata CSV files."""

from __future__ import annotations

import csv
import io
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


def apply_augmentations(
    image: Image.Image,
    rng: random.Random,
    horizontal_flip_prob: float,
    rotation_degrees: float,
    brightness_jitter: float,
    contrast_jitter: float,
) -> Image.Image:
    if horizontal_flip_prob > 0 and rng.random() < horizontal_flip_prob:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if rotation_degrees > 0:
        angle = rng.uniform(-rotation_degrees, rotation_degrees)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR)

    if brightness_jitter > 0:
        brightness_factor = rng.uniform(1 - brightness_jitter, 1 + brightness_jitter)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

    if contrast_jitter > 0:
        contrast_factor = rng.uniform(1 - contrast_jitter, 1 + contrast_jitter)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    return image


class PKLotTarDataset(Dataset):
    """Read PKLot images from the tar archive or extracted image files."""

    def __init__(
        self,
        archive_path: str | Path,
        metadata_csv: str | Path,
        extracted_root: str | Path | None = None,
        image_size: int = 128,
        limit: int | None = None,
        seed: int = 42,
        train: bool = False,
        augmentation: dict | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.archive_path = Path(archive_path)
        self.metadata_csv = Path(metadata_csv)
        self.extracted_root = Path(extracted_root) if extracted_root else None
        self.image_size = image_size
        self.seed = seed
        self.train = train
        self.augmentation = augmentation or {}
        self.return_metadata = return_metadata
        self.samples = self._load_samples(limit)
        self._archive: tarfile.TarFile | None = None

    def _load_samples(self, limit: int | None) -> list[dict[str, str]]:
        with self.metadata_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            samples = list(reader)
        if limit is not None:
            rng = random.Random(self.seed)
            rng.shuffle(samples)
            samples = samples[:limit]
        return samples

    def _get_archive(self) -> tarfile.TarFile:
        if self._archive is None:
            self._archive = tarfile.open(self.archive_path, "r")
        return self._archive

    def _open_image(self, row: dict[str, str]) -> Image.Image:
        if self.extracted_root is not None:
            file_path = self.extracted_root / row["archive_member"]
            return Image.open(file_path).convert("RGB")

        archive = self._get_archive()
        member_handle = archive.extractfile(row["archive_member"])
        if member_handle is None:
            raise FileNotFoundError(f"Could not read {row['archive_member']} from archive.")

        image_bytes = member_handle.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples[index]
        image = self._open_image(row)

        if self.train:
            image = apply_augmentations(
                image=image,
                rng=random.Random(self.seed + index),
                horizontal_flip_prob=self.augmentation.get("horizontal_flip_prob", 0.0),
                rotation_degrees=self.augmentation.get("rotation_degrees", 0.0),
                brightness_jitter=self.augmentation.get("brightness_jitter", 0.0),
                contrast_jitter=self.augmentation.get("contrast_jitter", 0.0),
            )

        image = image.resize((self.image_size, self.image_size))

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array)

        # Normalize to roughly centered image channels without requiring torchvision.
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) / std

        label = torch.tensor(int(row["label_id"]), dtype=torch.long)
        if self.return_metadata:
            return tensor, label, row
        return tensor, label
