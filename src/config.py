MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

MAX_LENGTH = 128

LEARNING_RATE = 2e-5

BATCH_SIZE = 8

NUM_EPOCHS = 2

OUTPUT_DIR = "./results"

MODEL_SAVE_PATH = "./models/sentiment_model"

# Optional dataset limits for faster development and CPU-only execution.
MAX_TRAIN_SAMPLES = 1000
MAX_VAL_SAMPLES = 250

# Optional limits for faster CPU training during development.
MAX_TRAIN_SAMPLES = 1000
MAX_VAL_SAMPLES = 250

# Monitoring: sample sizes for reference (validation split) and
# current/production (test split) datasets used by Evidently reports.
MAX_REFERENCE_SAMPLES = 200
MAX_CURRENT_SAMPLES = 200

MONITORING_DATA_DIR = "monitoring/data"
MONITORING_REPORTS_DIR = "monitoring/reports"