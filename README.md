# Smart Parking Camera Project

This project detects whether a parking space is `Empty` or `Occupied` using the PKLot dataset.

## Project stages

1. Dataset setup and split generation
2. Preprocessing and augmentation
3. Baseline occupancy detection model
4. System-level output and visualization
5. Robustness analysis and reporting

## Current status

The repository currently includes:

- `PKLot.tar`: raw dataset archive
- `scripts/prepare_dataset.py`: dataset indexing and train/validation/test split generation
- `scripts/extract_dataset.py`: optional extraction utility for faster file-based training
- `configs/baseline.yaml`: baseline experiment configuration
- `data/`: generated metadata files

## Quick start

Create the metadata split files from the archive:

```bash
python3 scripts/prepare_dataset.py \
  --archive PKLot.tar \
  --output-dir data/metadata \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

This does not extract the whole dataset. It scans the archive and creates CSV files pointing at each image path inside the tarball.

Optionally extract one or more splits to regular image files:

```bash
python3 scripts/extract_dataset.py \
  --archive PKLot.tar \
  --metadata-dir data/metadata \
  --output-dir data/extracted \
  --splits train val test
```

Then set `data.extracted_root: data/extracted` in the config to train from files instead of directly from the tar archive.

## Planned next steps

- Train a transfer-learning baseline
- Add evaluation plots and a simple occupancy visualization
