import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from scipy.special import softmax
from src.config import MODEL_SAVE_PATH


tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SAVE_PATH
)


def predict(text):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():
        outputs = model(**encoded)

    scores = softmax(
        outputs.logits.numpy()[0]
    )

    prediction = scores.argmax()

    labels = {

        0: "negative",

        1: "neutral",

        2: "positive",
    }

    return {

        "label": labels[prediction],

        "confidence": float(scores[prediction]),

        "scores": scores.tolist(),
    }


if __name__ == "__main__":

    text = "I really love this new product!"

    print(predict(text))