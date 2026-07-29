import os
from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition, MulticlassClassification, Report
from evidently.presets import ClassificationPreset, DataDriftPreset

from src.config import MONITORING_DATA_DIR, MONITORING_REPORTS_DIR

ACCURACY_DROP_THRESHOLD = 0.1


def _load(name):
    return pd.read_csv(Path(MONITORING_DATA_DIR) / name)


def _accuracy(df):
    return (df["target"] == df["prediction"]).mean()


def _data_definition():
    return DataDefinition(
        numerical_columns=["confidence", "text_length"],
        classification=[
            MulticlassClassification(
                target="target",
                prediction_labels="prediction",
            )
        ],
    )


def main():
    reference_df = _load("reference.csv")
    current_df = _load("current.csv")

    data_definition = _data_definition()
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report(metrics=[ClassificationPreset(), DataDriftPreset()])
    snapshot = report.run(current_data=current_dataset, reference_data=reference_dataset)

    os.makedirs(MONITORING_REPORTS_DIR, exist_ok=True)
    snapshot.save_html(str(Path(MONITORING_REPORTS_DIR) / "report.html"))
    snapshot.save_json(str(Path(MONITORING_REPORTS_DIR) / "report.json"))

    reference_accuracy = _accuracy(reference_df)
    current_accuracy = _accuracy(current_df)
    drop = reference_accuracy - current_accuracy

    print(f"[monitoring] reference accuracy: {reference_accuracy:.3f}")
    print(f"[monitoring] current accuracy: {current_accuracy:.3f}")

    if drop > ACCURACY_DROP_THRESHOLD:
        print(
            f"[monitoring] WARNING: accuracy dropped by {drop:.3f}, "
            f"above threshold {ACCURACY_DROP_THRESHOLD}"
        )
    else:
        print(f"[monitoring] accuracy within threshold (drop={drop:.3f})")


if __name__ == "__main__":
    main()
