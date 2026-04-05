from datasets import load_dataset, Dataset, interleave_datasets

# Define a hardcoded dataset of strings (streaming=False)
hardcoded_data = {
    "text": [
        "This is the first example.",
        "Here is another sentence.",
        "This is the third example.",
        "Adding more examples to the dataset."
    ]
}
# Convert the hardcoded data into a Hugging Face Dataset
hardcoded_dataset = Dataset.from_dict(hardcoded_data).with_format("torch")

# Load SlimPajamas dataset with streaming=True
slim_pajamas_dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True).with_format("torch")

# Interleave the datasets with equal probability
train_dataset = interleave_datasets([slim_pajamas_dataset, hardcoded_dataset], probabilities=[0.5, 0.5])

# Example: Iterating through the combined dataset
for idx, batch in enumerate(train_dataset):
    print(f"Batch {idx}: {batch['text']}")
    # Your training code here

    # Break after a few iterations for demonstration purposes
    if idx > 5:
        break
