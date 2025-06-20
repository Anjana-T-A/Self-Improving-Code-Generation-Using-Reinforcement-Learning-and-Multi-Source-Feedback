import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl.evaluate_model import evaluate_pass_at_k
from data.datasets import load_datasets


datasets = load_datasets()
model_checkpoint = "ppo_checkpoints"

model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"  # Change for each model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metrics = evaluate_pass_at_k(model, tokenizer, datasets["test"], device, k_values=[1,5,10])
print(metrics)