# MedicalAI v2 Training Engine Upgrade - TODO

## Step 1: Audit & planning confirmation
- [x] Read current training wiring (train.py, trainer.py, losses.py, config.py, checkpoint.py)
- [x] Produce edit plan matching requested features

## Step 2: Implement loss factory
- [ ] Extend `training/losses.py` with Dice/CE/combined/weighted CE/Focal/Tversky implementations
- [ ] Add configurable `build_loss_from_config(cfg)` factory (defaults match existing CombinedLoss(dice_weight=1.0))



## Step 3: Implement optimizer factory wiring
- [ ] Update `training/train.py` to use optimizer factory driven by config (Adam/AdamW/SGD)

## Step 4: Implement scheduler factory wiring
- [ ] Add scheduler builder driven by config for: None, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
- [ ] Update `training/trainer.py` scheduler stepping logic (metric stepping + per-batch for OneCycleLR)

## Step 5: Add gradient clipping
- [ ] Update `training/trainer.py` to clip gradients before optimizer step (configurable)

## Step 6: Improve TensorBoard logging
- [ ] Update `training/trainer.py` to log richer scalars per class and config text/hparams

## Step 7: Improve checkpoint saving & resume
- [ ] Update `training/utils/checkpoint.py` to save/load AMP scaler state and checkpoint schema version
- [ ] Update `training/trainer.py` resume path to restore scaler state

## Step 8: Extend config schema
- [ ] Update `training/config.py` with new loss/scheduler/grad-clip fields and defaults
- [ ] Ensure backward-compatible JSON load/save (new fields optional)

## Step 9: Wire everything in train entrypoint
- [ ] Update `training/train.py` to use factories for loss/optimizer/scheduler and pass training config into trainer

## Step 10: Smoke test
- [ ] Run `python -m training.trainer` self-test
- [ ] Run a minimal import check for `training/train.py`


