import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

MODEL_ID = "google-bert/bert-base-uncased"
MODEL_PATH = "google-bert/bert-base-uncased"  # Gets changed with tuned model path
TEST_DATASET_PATH = "test.jsonl"
BATCH_SIZE = 128
TOKENIZED_DATASET_PATH = "tokenized_test_dataset"

torch.set_float32_matmul_precision("high")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


if os.path.exists(TOKENIZED_DATASET_PATH):
    print("Loading existing tokenized dataset...")
    tokenized_test_dataset = load_from_disk(TOKENIZED_DATASET_PATH)
else:
    print("Tokenizing dataset...")
    train_dataset = load_dataset("json", data_files=TEST_DATASET_PATH)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    if "label" in train_dataset["train"].features.keys():
        train_dataset = train_dataset.rename_column("label", "labels")

    tokenized_test_dataset = train_dataset.map(
        tokenize, batched=True, remove_columns=["text"]
    )

    tokenized_test_dataset = train_dataset.map(
        tokenize, batched=True, remove_columns=["text"]
    )

    tokenized_test_dataset.save_to_disk(TOKENIZED_DATASET_PATH)
    print(f"Tokenized dataset saved at {TOKENIZED_DATASET_PATH}")


tokenized_test_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

num_labels = 5
label2id = {str(i): i for i in range(num_labels)}
id2label = {i: str(i) for i in range(num_labels)}

test_dataloader = DataLoader(
    tokenized_test_dataset["train"],
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers=8,
)

device = torch.device("cuda")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=num_labels, label2id=label2id, id2label=id2label
).to("cuda")

all_preds, all_labels = [], []

model.eval()

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        print(f"Batch: {i}/{len(test_dataloader)}")
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted"
)

classification_rep = classification_report(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_rep)
