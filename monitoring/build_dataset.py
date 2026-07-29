import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.config import (
    MAX_CURRENT_SAMPLES,
    MAX_REFERENCE_SAMPLES,
    MONITORING_DATA_DIR,
)
from src.inference import predict


def _build_split(split_name, max_samples):
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split_name)
    label_names = dataset.features["label"].names

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    rows = []
    for example in dataset:
        text = example["text"]
        result = predict(text, log=False)

        rows.append({
            "text": text,
            "text_length": len(text),
            "target": label_names[example["label"]],
            "prediction": result["label"],
            "confidence": result["confidence"],
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs(MONITORING_DATA_DIR, exist_ok=True)

    reference_df = _build_split("validation", MAX_REFERENCE_SAMPLES)
    reference_df.to_csv(
        Path(MONITORING_DATA_DIR) / "reference.csv", index=False
    )

    current_df = _build_split("test", MAX_CURRENT_SAMPLES)
    current_df.to_csv(
        Path(MONITORING_DATA_DIR) / "current.csv", index=False
    )

    print(f"[monitoring] reference: {len(reference_df)} rows")
    print(f"[monitoring] current: {len(current_df)} rows")


if __name__ == "__main__":
    main()
