# Smart Parking Camera Project Report

## 1. Problem formulation

The goal of this project is to detect whether a parking space is `Empty` or `Occupied` from images. This is a binary image classification task with practical real-world value for smart parking systems.

The original project dataset is the PKLot parking occupancy dataset. Because the full dataset is very large and the available machine is CPU-only, the final experiments were performed on a reproducible subset of the data. This choice keeps the workflow practical while still allowing a full machine learning pipeline, robustness analysis, and a system-level output demo.

## 2. Dataset setup

The project uses the segmented PKLot parking-slot images stored in `PKLot.tar`. These images are already labeled as `Empty` or `Occupied`, which makes them suitable for direct classification.

Full metadata indexing was performed first, producing the following split sizes:

| Split | Count | Empty | Occupied |
|---|---:|---:|---:|
| Train | 487,093 | 250,648 | 236,445 |
| Validation | 104,374 | 53,709 | 50,665 |
| Test | 104,384 | 53,714 | 50,670 |

For actual training and evaluation, a smaller extracted subset was used:

| Split | Count used in baseline experiment |
|---|---:|
| Train | 1,000 |
| Validation | 1,000 |
| Test | 1,000 |

The subset was sampled reproducibly with a fixed random seed, so the experiment can be repeated.

## 3. Preprocessing and augmentation

Each parking-slot image was:

- resized to `128 x 128`
- converted to RGB
- normalized with standard channel-wise mean and standard deviation

To improve generalization, training images used light augmentation:

- random horizontal flip
- small random rotation (`+-10` degrees)
- brightness jitter
- contrast jitter

These augmentations are modest enough to preserve the parking-slot content while still improving robustness.

## 4. Model and training setup

The baseline model is a compact custom CNN designed to run on CPU:

- 4 convolutional blocks
- batch normalization
- ReLU activations
- max pooling
- adaptive average pooling
- fully connected classifier with dropout

Training details:

- optimizer: Adam
- learning rate: `0.001`
- weight decay: `0.0001`
- batch size: `64`
- maximum epochs: `5`
- early stopping patience: `2`

The best model checkpoint was saved based on validation F1 score.

## 5. Baseline results

The main subset experiment produced the following learning curve:

| Epoch | Validation Accuracy | Validation F1 |
|---|---:|---:|
| 1 | 0.743 | 0.6445 |
| 2 | 0.867 | 0.8411 |
| 3 | 0.968 | 0.9665 |
| 4 | 0.947 | 0.9476 |
| 5 | 0.984 | 0.9833 |

Best validation result:

- best epoch: `5`
- best validation F1: `0.9833`

Final test performance:

| Metric | Value |
|---|---:|
| Loss | 0.0573 |
| Accuracy | 0.9850 |
| Precision | 0.9880 |
| Recall | 0.9821 |
| F1 score | 0.9851 |

These results show that even the lightweight baseline model performs very strongly on the chosen subset.

## 6. Robustness analysis

To go beyond a single overall accuracy number, the trained model was evaluated across different parking sites and weather conditions.

### By site

| Site | Count | Accuracy | F1 |
|---|---:|---:|---:|
| PUC | 609 | 0.9918 | 0.9914 |
| UFPR04 | 148 | 0.9797 | 0.9771 |
| UFPR05 | 243 | 0.9712 | 0.9761 |

### By weather

| Weather | Count | Accuracy | F1 |
|---|---:|---:|---:|
| Cloudy | 337 | 0.9970 | 0.9963 |
| Rainy | 165 | 0.9758 | 0.9813 |
| Sunny | 498 | 0.9799 | 0.9808 |

### Interpretation

- The model performs well across all measured sites and weather conditions.
- The best results appeared in `Cloudy` conditions.
- `Rainy` and `UFPR05` were slightly more difficult, but performance remained strong.
- This suggests the model is not overfitting to only one environment within the chosen subset.

## 7. System-level output

A simple system-output demo was created to simulate a smart parking occupancy board. The system:

- samples parking-slot images
- predicts `Empty` or `Occupied`
- shows color-coded outputs
- displays confidence values
- reports the predicted occupancy ratio for the current board

This provides a practical visual layer on top of the classifier and makes the model output easier to interpret in an application context.

Generated artifact:

- `outputs/baseline_subset_1k/system_output/occupancy_board.png`

## 8. Limitations

This project has several important limitations:

- The final training was performed on a subset, not the entire PKLot dataset.
- The machine used for development was CPU-only, which limited large-scale experimentation.
- The system-output demo works on segmented parking-slot images, not full parking-lot scenes with live slot detection.
- A stronger model such as transfer learning with a pretrained backbone could likely improve robustness further.

These limitations should be stated clearly in the final presentation or submission.

## 9. Conclusion

This project successfully built a complete parking occupancy detection pipeline:

- dataset preparation
- preprocessing and augmentation
- baseline CNN training
- robustness analysis
- visual system output

Despite using a subset of the original PKLot dataset, the baseline achieved excellent results with `98.5%` test accuracy and `0.9851` test F1 score. The project therefore demonstrates that parking occupancy detection can be solved effectively with a relatively simple deep learning approach, even under local hardware constraints.

## 10. Reproducibility

Main files used in the project:

- `scripts/prepare_dataset.py`
- `scripts/extract_dataset.py`
- `scripts/train_baseline.py`
- `scripts/robustness_analysis.py`
- `scripts/system_output_demo.py`
- `configs/baseline_subset_1k.yaml`

Main outputs:

- `outputs/baseline_subset_1k/metrics.json`
- `outputs/baseline_subset_1k/robustness/summary.json`
- `outputs/baseline_subset_1k/robustness/by_site.csv`
- `outputs/baseline_subset_1k/robustness/by_weather.csv`
- `outputs/baseline_subset_1k/system_output/occupancy_board.png`
