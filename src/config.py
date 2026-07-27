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