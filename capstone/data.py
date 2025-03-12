from datasets import load_dataset

dataset_name = "Yelp/yelp_review_full"
sample_size = 200000
test_size = 0.3  # 30% test split

full_dataset = load_dataset(dataset_name, split="train")

# Randomly pick sample_size records
sampled_dataset = full_dataset.shuffle(seed=69).select(range(sample_size))

split_dataset = sampled_dataset.train_test_split(test_size=test_size)

split_dataset["train"].to_json("train.jsonl")
split_dataset["test"].to_json("test.jsonl")

print(f"Sampled dataset saved: {sample_size} rows total")
print(f"Train dataset: {len(split_dataset['train'])} rows")
print(f"Test dataset: {len(split_dataset['test'])} rows")
