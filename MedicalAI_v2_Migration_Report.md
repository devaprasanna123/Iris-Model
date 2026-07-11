# MedicalAI v2 Migration Report (Architecture Planning Only)

## Scope & Non-Goals
- **Migration only**: Upgrade architecture for **MedicalAI v2** while preserving existing working behavior.
- **No code changes**: This report proposes structure, modules, and upgrade order only.
- **Preserve stable contracts**: Tensor shapes and filesystem conventions used by the existing pipeline.

---

## 1) Project Overview
**MedicalAI v1** is a dataset engineering + PyTorch training pipeline for **U-Net semantic segmentation** on AS-OCT imagery.

### Medical task assumptions / outputs
- Semantic segmentation into **3 classes**:
  - Background = 0
  - Cornea = 1
  - Iris = 2

### Included subsystems (v1)
1. Dataset Loader + Dataset (`OctDataset`) and DataLoader factories
2. Configuration (`TrainingConfig`) with JSON save/load
3. Logger (`Logger`) and Checkpoint manager (`CheckpointManager`)
4. Training engine (`Trainer`) with:
   - AMP mixed precision
   - metric computation (Dice/IoU/Pixel Accuracy/Precision/Recall/F1)
   - early stopping
   - best/last checkpoint saving
   - TensorBoard optional logging
5. Evaluation entrypoint (`training/evaluate.py`)
6. Prediction entrypoint (`training/predict.py`)
7. Dataset engineering / Phase 1 scripts (verification, merge, LabelMe conversion, statistics, split, QC reports)
8. Google Colab support via path auto-detection in config

---

## 2) Architecture Overview (v1)

### 2.1 Training module layout (`MedicalAI/training/`)
- `config.py`
  - `TrainingConfig` dataclass tree (dataset, checkpoint paths, logging dirs, classes, hyperparams, image params)
  - device auto-detection helper
  - JSON serialization helpers
- `dataloaders.py`
  - `create_train_loader`, `create_val_loader`, `create_test_loader`
  - uses `OctDataset`
- `datasets/oct_dataset.py`
  - `OctDataset`:
    - fixed resize to **512×512**
    - image: BMP/PNG read → resize INTER_LINEAR → float32 /255 → tensor (3,H,W)
    - mask: PNG read (grayscale) → resize INTER_NEAREST → int64 tensor (H,W)
- `models/unet.py`
  - UNet producing **logits** `(B,3,H,W)` (no softmax)
- `losses.py`
  - `DiceLoss` (softmax internally)
  - `CombinedLoss` = CE + dice_weight * DiceLoss
- `metrics.py`
  - `MetricsSpec` and metric functions operating on logits or labels
  - computes per-class and mean Dice/IoU, plus pixel accuracy, precision/recall/F1
- `utils/logger.py`
  - `Logger` wrapper around stdlib logging with file + console handlers
- `utils/checkpoint.py`
  - `CheckpointManager`:
    - saves `best` and `last` as single checkpoint files
    - stores metadata (`epoch`, losses, dice, best_dice, training_config, timestamp)
    - loads model/optimizer/scheduler state dicts as provided
- `trainer.py`
  - `Trainer.fit()`:
    - train loop + val loop
    - AMP support and scaler
    - early stopping based on val dice
    - saves `last` every epoch, `best` on improvement
    - scheduler stepping per epoch
- Entrypoints:
  - `train.py` (main training)
  - `evaluate.py` (main evaluation)
  - `predict.py` (inference CLI)

### 2.2 Dataset engineering / Phase 1 scripts (repo root)
- `phase1_runner.py` orchestrates subprocess steps:
  - merge dataset sources
  - convert LabelMe → masks
  - visualize masks overlays
  - compute dataset statistics
  - split into train/val/test
  - run strict mask verification and generate Phase 1 completion report

---

## 3) Dependency Graph (conceptual)

### Runtime (training/evaluation)
- `train.py` → `TrainingConfig` → `dataloaders.py` → `OctDataset`
- `train.py` → `UNet` + `CombinedLoss` + `MetricsSpec`
- `train.py` → `Trainer` + `Logger` + `CheckpointManager`
- `Trainer` → metrics + losses + checkpoint save/load

### Inference
- `predict.py` → `TrainingConfig` → `UNet` + `CheckpointManager`
- `predict.py` → OpenCV preprocessing → model logits → argmax labels
- `predict.py` → colorization + overlay output artifacts

### Offline dataset engineering
- `phase1_runner.py` → subprocess calls to scripts under `MedicalAI/` root
- QC scripts → reports/visualization artifacts under dataset folders

---

## 4) Reusable Components (should remain untouched)
These modules are stable “building blocks” and should be reused by v2:
- `training/config.py` (reuse; optionally wrap/validate later)
- `training/datasets/oct_dataset.py` (dataset contract)
- `training/dataloaders.py` (loader factories)
- `training/models/unet.py` (model forward contract)
- `training/losses.py` (loss contract)
- `training/metrics.py` (metrics contract)
- `training/utils/logger.py` (logging contract)
- `training/utils/checkpoint.py` (checkpoint payload contract)
- `training/trainer.py` (training harness contract)

---

## 5) Components to Upgrade for MedicalAI v2

### 5.1 Import structure / packaging
- Current v1 code uses imports like `from training.config import TrainingConfig`.
- v2 should provide a **proper Python package layout** (e.g., `medicalai/`) with:
  - production-grade module boundaries
  - backward-compatible shims so v1 entrypoints keep working

### 5.2 Entrypoints unification (thin facade)
- v1 has separate scripts: `train.py`, `evaluate.py`, `predict.py`.
- v2 should add a standardized CLI facade (wrapping existing entrypoints):
  - `medicalai train`
  - `medicalai eval`
  - `medicalai predict`

### 5.3 Configuration hardening (recommended, not implemented)
- v1 config auto-detects dataset roots for Colab; convenient but not production deterministic.
- v2 should introduce configuration “profiles” / validation layers (recommended later):
  - explicit dataset root
  - explicit checkpoint/log/output directories
  - validation of required folder layout

### 5.4 Contract documentation & tests (recommended)
- v2 should formalize and test core contracts:
  - dataset output shapes/dtypes
  - model output shapes
  - loss/metrics end-to-end on synthetic data
  - checkpoint save/load smoke test

### 5.5 Artifact conventions
- Ensure consistency of inference preprocessing vs dataset preprocessing:
  - v1 dataset resizing/normalization vs v1 `predict.py` OpenCV preprocessing
- v2 should centralize contract docs so both training and inference remain aligned.

---

## 6) New Components Required (v2)
Proposed additions (wrappers/adapters; no rewrites):
1. **v2 package skeleton**
   - `MedicalAI/src/medicalai/` (or similar)
2. **CLI facade module**
   - delegates to v1 `training/train.py`, `training/evaluate.py`, `training/predict.py`
3. **Builders / adapters**
   - thin wrappers that call v1 constructors (model/data/loss/metrics/trainer)
4. **Contract test module**
   - synthetic tests for dataset/model/loss/metrics/checkpoint
5. **Contract + constants docs**
   - single source of truth for label mapping, resize size, normalization rules

---

## 7) Deprecated Modules (or lower priority)
No “hard deprecated” modules can be concluded from inspection alone.
Migration should treat these as **lower priority** until after v2 facade stabilizes:
- Phase 1 orchestration scripts remain as offline tools.
- `phase1_runner.py` can be wrapped later by v2 CLI (optional), but it is not required for production training/inference.

---

## 8) Risk Analysis

### High Risks
1. **Breaking tensor contracts**
   - Any change to shapes/dtypes/label encoding impacts dice/CE/metrics and model training quality.
2. **Breaking import paths / runtime entrypoints**
   - v1 currently runs assuming the `training/` directory is importable.
3. **Preprocessing mismatch (training vs inference)**
   - `predict.py` uses OpenCV preprocessing; v2 must preserve parity with `OctDataset`.
4. **Checkpoint schema compatibility**
   - `CheckpointManager` stores payload keys and metadata; extending schema must be versioned.

### Medium Risks
5. **Filesystem layout assumptions**
   - dataset split folder layout and filename/stem pairing.
6. **Device and AMP behavior**
   - Trainer uses AMP conditional logic; wrappers should pass through flags unchanged.

### Low Risks
7. Logger/TensorBoard optional behavior can be standardized but not required for core migration.

---

## 9) Migration Strategy (No code changes)

### Guiding principles
- **Facade + Adapter migration**: v2 adds production structure while delegating to v1 stable modules.
- **Non-destructive**: keep v1 code runnable.
- **Incremental**: introduce one layer at a time (package → CLI facade → tests → production hardening).

### Phased plan
**Phase A — v2 scaffolding (no behavioral change):**
1. Create v2 package skeleton with namespace `medicalai`.
2. Add import shims/wrappers so existing v1 entrypoints remain compatible.

**Phase B — v2 CLI facade:**
3. Provide `medicalai train/eval/predict` commands that call v1 entrypoints.

**Phase C — contract tests & documentation:**
4. Add synthetic tests and contract docs to prevent future mismatches.

**Phase D — configuration hardening (later):**
5. Add optional config validation and profiles; keep v1 defaults working.

---

## 10) Recommended Development Order (v2)
1. **Complete repo audit**: map entrypoints and import assumptions.
2. **Scaffold v2 package structure** (no runtime changes).
3. **Add CLI facade wrappers** delegating to v1 `train/evaluate/predict`.
4. **Add builders/adapters** (thin delegations) to centralize wiring.
5. **Add contract tests** (synthetic dataset/model/loss/metrics/checkpoint).
6. **Verify preprocessing parity**: dataset vs predict input pipeline produces same tensor contracts.
7. **Only after stability**: consider config validation profiles and cleanup.

---

## 11) Recommended Folder Structure (v2)

Proposed structure:
```text
MedicalAI/
  src/medicalai/
    __init__.py
    cli/
      main.py                  # command dispatcher
    training/
      entrypoints.py          # delegates to v1 train/eval/predict
      builders.py             # thin adapters
      contracts.py            # label map + preprocessing contract docs
    tests/
      test_contracts.py      # synthetic + smoke tests
  training/                   # keep v1 stable code untouched
  dataset/                    # excluded data layout stays as-is
  requirements.txt
```

Compatibility approach:
- Keep current `MedicalAI/training/*.py` functional.
- v2 package uses wrappers around v1 modules.

---

## 12) Technical Recommendations (architecture)
- **Do not rewrite**: keep UNet, dataset, losses, metrics, trainer.
- Introduce **interfaces** (docs + types) to codify:
  - label encoding (0/1/2)
  - input normalization ([0,1] float32)
  - resize rules (512×512; INTER_LINEAR for image; INTER_NEAREST for masks)
- Add a small internal “contract test” suite to enforce invariants.
- Extend checkpointing only via versioned metadata when needed.

---

## 13) Deliverable Confirmation
This migration report is planning-only:
- No code modifications were made.
- Recommended folder structure and upgrade order are provided.

