import os

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.config import (
    MODEL_NAME,
    OUTPUT_DIR,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    MODEL_SAVE_PATH,
)

from src.preprocessing import tokenizer

from src.evaluation import compute_metrics


def train(dataset):

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
    )

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        greater_is_better=True,

        dataloader_pin_memory=False,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()

    print(metrics)

    trainer.save_model(MODEL_SAVE_PATH)

    tokenizer.save_pretrained(MODEL_SAVE_PATH)