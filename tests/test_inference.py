import pytest

from src.config import MODEL_SAVE_PATH
import os

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_SAVE_PATH),
    reason="Trained model not found, run train.py first",
)


def test_predict_returns_valid_sentiment():
    from src.inference import predict

    result = predict("I really love this new product!", log=False)

    assert result["label"] in {"negative", "neutral", "positive"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["scores"]) == 3
    assert abs(sum(result["scores"]) - 1.0) < 1e-3


def test_predict_handles_negative_text():
    from src.inference import predict

    result = predict("This is terrible, I hate it.", log=False)

    assert result["label"] in {"negative", "neutral", "positive"}
    assert isinstance(result["confidence"], float)
