import pytest
import torch
import torch.nn as nn

from training.config import TrainingConfig
from training.loss_factory import create_loss
from training.optimizer_factory import create_optimizer
from training.scheduler_factory import create_scheduler


def _dummy_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_default_class_weights_are_configurable():
    cfg = TrainingConfig()
    assert cfg.training.weighted_ce_class_weights == (0.1, 1.0, 4.0)


def test_combined_loss_uses_configured_class_weights():
    cfg = TrainingConfig()
    cfg.training.loss_name = "dice_cross_entropy"
    loss = create_loss(cfg)

    assert hasattr(loss, "ce")
    assert loss.ce.weight is not None
    assert torch.allclose(loss.ce.weight, torch.tensor([0.1, 1.0, 4.0], dtype=torch.float32))


@pytest.mark.parametrize("opt_name", ["adam", "adamw", "sgd"])
def test_create_optimizers(opt_name):
    cfg = TrainingConfig()
    cfg.training.optimizer = opt_name
    model = _dummy_model()

    opt = create_optimizer(cfg, model)
    assert isinstance(opt, torch.optim.Optimizer)


@pytest.mark.parametrize("sched_name", [
    "none",
    "cosine",
    "cosineannealingwarmrestarts",
    "reducelronplateau",
    "onecyclelr",
])
def test_create_schedulers(sched_name):
    cfg = TrainingConfig()
    cfg.training.scheduler = sched_name
    model = _dummy_model()
    opt = create_optimizer(cfg, model)

    # OneCycleLR requires steps_per_epoch; provide a reasonable default for tests
    sched = create_scheduler(cfg, opt, steps_per_epoch=5)
    if sched_name == "none":
        assert sched is None
    else:
        # Scheduler should expose step method
        assert hasattr(sched, "step")
*** End Patch