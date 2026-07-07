# MedicalAI - TODO

- [ ] Update `MedicalAI/training/train.py` to remove verification-only single-epoch mode.
- [ ] Ensure training always uses `cfg.training.epochs` (no hardcoded 1, no verification logger message).
- [ ] Replace output message `Starting 1-epoch verification training...` with `Starting full training...`.
- [ ] Add required configuration banner: device, dataset root, batch size, epochs, learning rate, optimizer, scheduler.
- [ ] Ensure trainer runs for `for epoch in range(cfg.training.epochs)` or equivalent.
- [ ] Validate no remaining occurrences of verification/epoch=1 overrides in `train.py`.
- [ ] Commit and push changes.

