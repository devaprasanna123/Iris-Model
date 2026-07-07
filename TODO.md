# TODO (MedicalAI)

- [ ] Update `MedicalAI/training/train.py`:
  - [ ] Remove temporary early `return` after `===== DataLoader initialization: OK =====`
  - [ ] After DataLoader init: create UNet, move to device, create CombinedLoss
  - [ ] Create AdamW optimizer from config values
  - [ ] Create scheduler only if enabled in config
  - [ ] Print required training configuration block before training starts
  - [ ] Force exactly 1 epoch verification run
  - [ ] Initialize Trainer, call `trainer.fit()`
  - [ ] Ensure epoch-end metrics are printed and checkpoint saves to `best_model.pt` and `last_model.pt`
  - [ ] Ensure TensorBoard logs are written
  - [ ] After epoch success, print `✅ First training epoch completed successfully.`

