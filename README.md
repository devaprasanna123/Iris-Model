# MedicalAI — U-Net Semantic Segmentation (AS-OCT Cornea/Iris)

## Project Overview
MedicalAI is a dataset-preparation and training-ready codebase for **U-Net semantic segmentation** of AS-OCT imagery. The dataset pipeline verifies raw inputs, merges dataset sources, converts LabelMe polygon annotations into segmentation masks, computes dataset statistics, and produces train/val/test splits.

## Features
- Dataset verification and QC reporting
- Merge of full-frame + partial-frame datasets (duplicate-safe)
- LabelMe polygon → mask conversion
- Dataset statistics generation
- Train/val/test split (deterministic seed)

## Dataset
**Dataset content is intentionally excluded from this Git repository** due to size.

Download it separately and place it under:

```text
MedicalAI/
    dataset/
        images/
        masks/
        train/
        val/
        test/
```

The project scripts expect the dataset directory layout above.

## Project Structure
```text
MedicalAI/
  training/
    config.py
    dataloaders.py
    losses.py
    metrics.py
    models/
      unet.py
    datasets/
      oct_dataset.py
    utils/
      logger.py
      checkpoint.py

  (dataset is excluded)
  dataset/
    images/
    annotations/
    masks/
    train/val/test/

  Dataset engineering scripts:
    phase1_runner.py
    verify_dataset.py
    merge_dataset.py
    convert_labelme_to_masks.py
    visualize_masks.py
    split_dataset.py
    dataset_statistics.py
    dataset_consistency_report.py

  requirements.txt
  .gitignore
  README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Training Roadmap
1. **Prepare dataset** (Phase 1)
   - Run the dataset verification, merge, conversion to masks, visualization QC, statistics, and split.
2. **Train U-Net**
   - Use the `training/` modules (config/dataloaders/model/loss/metrics) to train.
3. **Evaluate and Predict**
   - Use metrics + inference scripts (extend as needed).

## Current Progress
- Phase 1 dataset engineering scripts are included.
- Model training modules are included under `training/`.

## Future Work
- Add standardized training entrypoints (single CLI runner)
- Add inference script and export helpers
- Add experiment tracking integration (optional)

## Classes
- **Background**
- **Cornea**
- **Iris**

## Hardware Requirements
- Recommended: GPU (CUDA) for faster training
- Minimum: CPU-only works for testing/smoke runs (slower)

## Google Colab Support
You can run this project in Colab by:
1. Uploading the code
2. Downloading the dataset separately
3. Mounting dataset into `MedicalAI/dataset/`
4. Installing dependencies and running Phase 1 + training modules

## License
Specify your license (e.g., MIT/Apache-2.0).
