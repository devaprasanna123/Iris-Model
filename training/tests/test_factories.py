import pytest
import torch
import torch.nn as nn

from training.config import TrainingConfig
from training.optimizer_factory import create_optimizer
from training.scheduler_factory import create_scheduler


def _dummy_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


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