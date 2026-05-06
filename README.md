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

- `scripts/prepare_dataset.py`: dataset indexing and train/validation/test split generation
- `scripts/extract_dataset.py`: optional extraction utility for faster file-based training
- `configs/baseline_subset_70_15_15.yaml`: final subset experiment configuration
- `data/`: generated metadata files
- `outputs/baseline_subset_70_15_15/`: final experiment metrics, robustness outputs, and system demo

The raw `PKLot.tar` archive is not included in the final submission package. The submission keeps the code, metadata, final configuration, and final reported outputs needed to understand and reproduce the workflow when the dataset archive is available locally.

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

## Final experiment

The final reported subset experiment uses a reproducible `3000`-sample subset with the original split ratio preserved and semi-stratified by parking site and occupancy class:

- train: `2100`
- validation: `450`
- test: `450`

Main config:

```bash
configs/baseline_subset_70_15_15.yaml
```

Main outputs:

- `outputs/baseline_subset_70_15_15/metrics.json`
- `outputs/baseline_subset_70_15_15/robustness/summary.json`
- `outputs/baseline_subset_70_15_15/system_output/occupancy_board.png`
