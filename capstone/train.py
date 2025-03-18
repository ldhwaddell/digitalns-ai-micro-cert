import os
import torch

import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, load_from_disk

MODEL_ID = "google-bert/bert-base-uncased"
TRAIN_DATASET_PATH = "train.jsonl"
BATCH_SIZE = 128
TOKENIZED_DATASET_PATH = "tokenized_train_dataset"
OUTPUT_DIR = "finetuned_model"
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
PLOT_OUTPUT_PATH = "training_loss_plot.png"

torch.set_float32_matmul_precision("high")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if os.path.exists(TOKENIZED_DATASET_PATH):
    print("Loading existing tokenized dataset...")
    tokenized_train_dataset = load_from_disk(TOKENIZED_DATASET_PATH)
else:
    print("Tokenizing dataset...")
    train_dataset = load_dataset("json", data_files=TRAIN_DATASET_PATH)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    if "label" in train_dataset["train"].features.keys():
        train_dataset = train_dataset.rename_column("label", "labels")

    tokenized_train_dataset = train_dataset.map(
        tokenize, batched=True, remove_columns=["text"]
    )

    tokenized_train_dataset.save_to_disk(TOKENIZED_DATASET_PATH)
    print(f"Tokenized dataset saved at {TOKENIZED_DATASET_PATH}")

tokenized_train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

num_labels = 5
label2id = {str(i): i for i in range(num_labels)}
id2label = {i: str(i) for i in range(num_labels)}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=num_labels, label2id=label2id, id2label=id2label
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="no",  # No evaluation during training
    save_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    logging_dir="logs",
    logging_steps=10,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset["train"],
)

print("Starting training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
print(f"Model saved at {OUTPUT_DIR}")


# Extract and Save Loss Plot
losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
steps = list(range(1, len(losses) + 1))

plt.figure(figsize=(10, 5))
plt.plot(steps, losses, marker="o", linestyle="-", label="Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid()
plt.savefig(PLOT_OUTPUT_PATH)
print(f"Training loss plot saved at {PLOT_OUTPUT_PATH}")
