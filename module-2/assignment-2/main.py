from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import re
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(
        r"[^a-zA-Z0-9\s]", "", text
    )  # Remove punctuation and special characters
    return text


def task_1():
    dataset = load_dataset("imdb", split="train[:5000]")

    # Clean step
    dataset = dataset.map(lambda x: {"text": preprocess_text(x["text"])})

    # tokenize step
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"], padding="max_length", truncation=True, max_length=256
        ),
        batched=True,
    )

    # Convert the token ids returned from distilbert into actual text
    df = pd.DataFrame(
        {
            "original_text": dataset["text"],
            "tokenized_text": [
                tokenizer.convert_ids_to_tokens(t)
                for t in tokenized_dataset["input_ids"]
            ],
            "label": dataset["label"],
        }
    )

    # Print some random samples
    # iterrows is bad
    samples = df.sample(n=3, random_state=1)
    for _, row in samples.iterrows():
        print(f"Original: {row['original_text']}\n")
        print(f"Tokenized: {' '.join(row['tokenized_text'])}\n")
        print("=" * 80)

    return tokenized_dataset


accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def task_2(dataset):
    split = tokenized_dataset.train_test_split(test_size=0.2, seed=99)
    train_dataset = split["train"]
    test_dataset = split["test"]

    small_train_dataset = train_dataset.shuffle(seed=42).select(range(3000))
    small_eval_dataset = test_dataset.shuffle(seed=42).select(range(1000))

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./results",  # Directory to save model checkpoints
        evaluation_strategy="epoch",  # Evaluate after each epoch
        learning_rate=2e-5,  # Learning rate
        per_device_train_batch_size=8,  # Batch size for training
        per_device_eval_batch_size=8,  # Batch size for evaluation
        num_train_epochs=3,  # Number of epochs
        weight_decay=0.01,  # Weight decay for regularization
        save_strategy="epoch",  # Save model after each epoch
        logging_dir="./logs",  # Directory for logs
        logging_steps=10,  # Log every 10 steps
        report_to="none",  # Disable wandb logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    tokenized_dataset = task_1()
    task_2(tokenized_dataset)


"""
Evaluation results: {'eval_loss': 1.8635810192790814e-05, 'eval_accuracy': 1.0, 'eval_runtime': 21.1671, 'eval_samples_per_second': 47.243, 'eval_steps_per_second': 5.905, 'epoch': 3.0}
"""
