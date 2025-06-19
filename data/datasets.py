from datasets import load_dataset, DatasetDict

def load_datasets():
    datasets = DatasetDict()
    datasets['train'] = load_dataset("mbpp", split="train[:80%]")
    datasets['test'] = load_dataset("mbpp", split="train[80%:]")
    datasets['humaneval'] = load_dataset("openai_humaneval", split="test")
    return datasets