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

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from training.config import TrainingConfig
from training.dataloaders import create_test_loader
from training.evaluation import evaluate_model
from training.metrics import MetricsSpec
from training.models.model_factory import create_model
from training.utils.checkpoint import CheckpointManager
from training.utils.logger import Logger


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg = TrainingConfig()
    set_random_seed(int(cfg.training.seed))

    device = torch.device(cfg.device.device)

    logger = Logger(name="MedicalAI.evaluate", log_dir=cfg.logs.log_dir)
    logger.info("Evaluating on device=%s", device)

    model = create_model(cfg)
    model.to(device)

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

    test_loader = create_test_loader(
        dataset_path=cfg.dataset.dataset_root,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.workers),
        shuffle=False,
    )

    output_root = Path(cfg.logs.log_dir) / "evaluation"
    report = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        spec=cfg.classes and MetricsSpec(num_classes=cfg.classes.number_of_classes),
        output_root=output_root,
        checkpoint_metadata=metadata.to_dict(),
        top_k=20,
    )

    logger.info("Evaluation report saved to %s", output_root)
    logger.info("Overall metrics: %s", report.get("overall", {}))
    per_class = report.get("per_class", {}) or {}
    if per_class:
        logger.info("Per-class metrics:")
        for metric_name in ("dice", "iou", "precision", "recall", "f1"):
            metrics_block = per_class.get(metric_name, {}) or {}
            for class_name in metrics_block:
                if class_name == "mean":
                    continue
                logger.info(
                    "%s %s: %.6f",
                    class_name.title(),
                    metric_name.title(),
                    float(metrics_block[class_name]),
                )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.close()


if __name__ == "__main__":
    main()

