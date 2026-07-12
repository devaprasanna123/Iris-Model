import torch

from training.metrics import MetricsSpec, evaluate_all_metrics, flatten_per_class_metrics


def test_flatten_per_class_metrics_includes_class_specific_entries() -> None:
    logits = torch.tensor(
        [
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)

    metrics = evaluate_all_metrics(logits, target, spec=MetricsSpec(num_classes=3))
    flat = flatten_per_class_metrics(metrics)

    assert "background_dice" in flat
    assert "cornea_dice" in flat
    assert "iris_dice" in flat
    assert "background_iou" in flat
    assert "cornea_iou" in flat
    assert "iris_iou" in flat
