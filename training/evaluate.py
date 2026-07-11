"""MedicalAI evaluation entrypoint.

Responsibilities:
- Load best_model.pth
- Run on test loader
- Compute and print metrics:
  Dice, IoU, Pixel Accuracy, Precision, Recall, F1
- No visualization

Constraints honored:
- Only creates this file.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from training.config import TrainingConfig
from training.dataloaders import create_test_loader
from training.metrics import (
    MetricsSpec,
    dice_score,
    iou_score,
    pixel_accuracy,
    precision_score,
    recall_score,
    f1_score,
)
from training.models.model_factory import create_model
from training.utils.checkpoint import CheckpointManager
from training.utils.logger import Logger


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: Any, device: torch.device) -> Any:
    imgs, masks = batch
    return imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)


def main() -> None:
    cfg = TrainingConfig()
    set_random_seed(int(cfg.training.seed))

    device = torch.device(cfg.device.device)

    logger = Logger(name="MedicalAI.evaluate", log_dir=cfg.logs.log_dir)
    logger.info("Evaluating on device=%s", device)

    # Model
    model = create_model(cfg)
    model.to(device)

    # Load checkpoint (best)


    checkpoint_manager = CheckpointManager(
        checkpoint_dir=cfg.checkpoint.checkpoint_dir,
        best_model_name=cfg.checkpoint.best_model_name,
        last_model_name=cfg.checkpoint.last_model_name,
        device=device,
    )

    metadata, _extra = checkpoint_manager.load(
        model=model,
        optimizer=None,
        scheduler=None,
        which="best",
        strict=True,
    )
    logger.info("Loaded best checkpoint epoch=%s best_dice=%s", metadata.epoch, metadata.best_dice)

    # Test loader
    dataset_root = cfg.dataset.dataset_root
    test_loader = create_test_loader(
        dataset_path=dataset_root,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.workers),
        shuffle=False,
    )

    model.eval()

    metrics_spec = MetricsSpec(num_classes=cfg.classes.number_of_classes)

    dice_sum = 0.0
    iou_sum = 0.0
    acc_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    f1_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)

            dice = dice_score(logits, masks, spec=metrics_spec, input_is_logits=True)["mean"]
            iou = iou_score(logits, masks, spec=metrics_spec, input_is_logits=True)["mean"]
            acc = pixel_accuracy(logits, masks, spec=metrics_spec, input_is_logits=True)
            prec = precision_score(logits, masks, spec=metrics_spec, input_is_logits=True)["mean"]
            rec = recall_score(logits, masks, spec=metrics_spec, input_is_logits=True)["mean"]
            f1 = f1_score(logits, masks, spec=metrics_spec, input_is_logits=True)["mean"]

            dice_sum += float(dice)
            iou_sum += float(iou)
            acc_sum += float(acc)
            prec_sum += float(prec)
            rec_sum += float(rec)
            f1_sum += float(f1)
            num_batches += 1

    denom = max(1, num_batches)
    report = {
        "Dice": dice_sum / denom,
        "IoU": iou_sum / denom,
        "Pixel Accuracy": acc_sum / denom,
        "Precision": prec_sum / denom,
        "Recall": rec_sum / denom,
        "F1": f1_sum / denom,
    }

    print("\n===== Test Metrics (best_model.pt) =====")
    for k, v in report.items():
        print(f"{k:>16}: {v:.6f}")

    logger.info("Evaluation complete: %s", report)
    logger.close()


if __name__ == "__main__":
    main()

