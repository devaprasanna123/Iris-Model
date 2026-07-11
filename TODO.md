# MedicalAI - Phase 2 (Dataset Pipeline v1 -> v2)

- [x] Gather final code context for dataset pipeline contracts (oct_dataset, dataloaders, config) and ensure no changes outside dataset pipeline files.
- [ ] Implement `training/config.py` additions: preprocessing + augmentation config moved from hardcoded values.
- [ ] Upgrade `training/datasets/oct_dataset.py`:
  - [ ] Add dataset modes: training, validation, test, prediction, external.
  - [ ] Integrate Albumentations:
    - [ ] Deterministic medical preprocessing: Resize, Normalization, optional CLAHE, contrast/brightness/gamma, noise robustness.
    - [ ] Training-only augmentation: HorizontalFlip (optional gating), small rotation, shift/scale, brightness/contrast, Gaussian noise, blur, gamma.
    - [ ] Validation/Test/Prediction/External: no augmentation.
  - [ ] Ensure masks remain integer class IDs and geometric transforms are applied jointly image+mask.
  - [ ] Preserve backward compatibility: existing OctDataset(images_dir, masks_dir, ...) works.
- [ ] Update `training/dataloaders.py` to construct OctDataset with correct mode and new preprocessing config, without changing Trainer/Model code.
- [ ] Add internal self-tests in dataset module for tensor shapes/dtypes and mode behavior.
- [ ] Run a lightweight import/test command (where possible) to ensure no syntax/import errors.
- [ ] Final internal code review and return COMPLETE updated files.

