from transformers import AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess(example):
    text = example["text"]

    new_text = []

    for word in text.split():

        if word.startswith("@") and len(word) > 1:
            word = "@user"

        elif word.startswith("http"):
            word = "http"

        new_text.append(word)

    example["text"] = " ".join(new_text)

    return example


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )