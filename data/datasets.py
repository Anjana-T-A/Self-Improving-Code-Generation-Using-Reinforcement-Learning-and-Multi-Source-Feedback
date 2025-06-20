from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from utils.prompts import create_prompt

def load_datasets():
    datasets = DatasetDict()
    datasets['train'] = load_dataset("mbpp", split="train[:80%]")
    datasets['test'] = load_dataset("mbpp", split="train[80%:]")
    datasets['humaneval'] = load_dataset("openai_humaneval", split="test")
    return datasets

def get_dataloader(dataset_split, tokenizer, device, batch_size=2, shuffle=False):
    return DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, device),
        num_workers=0
    )

def collate_batch(batch, tokenizer, device):
    prompts = [create_prompt(item["text"], item["code"]) for item in batch]
    # Tokenize prompts with padding
    tokenizer.padding_side = "left"  # Ensure padding is on the left
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    return input_ids, attention_mask, prompts, batch
