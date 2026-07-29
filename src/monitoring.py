import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("monitoring/logs/predictions.jsonl")


def log_prediction(text, result):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "label": result["label"],
        "confidence": result["confidence"],
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
