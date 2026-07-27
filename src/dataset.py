from datasets import load_dataset
from src.config import MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES
from src.preprocessing import preprocess, tokenize


def prepare_dataset():
    dataset = load_dataset(
        "cardiffnlp/tweet_eval",
        "sentiment"
    )

    dataset = dataset.map(preprocess)

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    dataset = dataset.rename_column("label", "labels")

    if MAX_TRAIN_SAMPLES is not None:
        dataset["train"] = dataset["train"].select(
            range(min(MAX_TRAIN_SAMPLES, len(dataset["train"])))
        )

    if MAX_VAL_SAMPLES is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(MAX_VAL_SAMPLES, len(dataset["validation"])))
        )

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return dataset
