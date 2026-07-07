# TODO - MedicalAI Phase 1 + Phase 2 prep

## Phase 1 (finalize dataset)
- [x] Gather current dataset integrity state and scripts
- [ ] Step 1: Delete ONLY sample 401 from working dataset (images/401.bmp, annotations/401.json, masks/401.png if exists)
- [ ] Step 2: Verify dataset integrity after removal (no corrupted JSON, no missing masks, one-to-one stems)
- [ ] Step 3: Regenerate dataset_statistics.json
- [ ] Step 4: Regenerate split dataset (train/val/test = 80/10/10, seed=42)
- [ ] Step 5: Regenerate phase1_report.md with required fields
- [ ] Freeze validated dataset into dataset_v1/

## Phase 2 prep (no training)
- [ ] Recommend Python 3.11
- [ ] Create virtual environment instructions
- [ ] Requirements for PyTorch (CUDA), TorchVision, OpenCV, Albumentations, NumPy, (MONAI optional), TensorBoard, Matplotlib, tqdm
- [ ] Verify GPU availability and print CUDA/GPU/VRAM/torch.cuda.is_available()
- [ ] Recommend best segmentation architecture for AS-OCT (Cornea+Iris) for RTX 3050 Laptop 6GB
- [ ] Generate detailed training implementation roadmap (no training run)
- [x] Complete MedicalAI Phase 2 training pipeline: trainer, train entrypoint, evaluate, predict
- [ ] Verify syntax + run minimal self-tests
- [ ] Run git status / add / commit / push

