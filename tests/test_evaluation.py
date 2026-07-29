import numpy as np

from src.evaluation import compute_metrics


def test_compute_metrics_perfect_predictions():
    logits = np.array([
        [5, 0, 0],
        [0, 5, 0],
        [0, 0, 5],
    ])
    labels = np.array([0, 1, 2])

    metrics = compute_metrics((logits, labels))

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_compute_metrics_returns_expected_keys():
    logits = np.array([[1, 0, 0], [0, 1, 0]])
    labels = np.array([0, 0])

    metrics = compute_metrics((logits, labels))

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
