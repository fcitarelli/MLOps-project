import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return {

        "accuracy": accuracy_score(labels, predictions),

        "precision": precision_score(
            labels,
            predictions,
            average="macro",
        ),

        "recall": recall_score(
            labels,
            predictions,
            average="macro",
        ),

        "f1": f1_score(
            labels,
            predictions,
            average="macro",
        ),
    }