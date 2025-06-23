import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl.evaluate_model import evaluate_model
from rl.model_loader import load_checkpoint
from data.datasets import load_datasets


datasets = load_datasets()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_path = "./ppo_checkpoints/epoch_3"  # adjust to your latest epoch folder
model_path = "./ppo_unit_test/epoch_3"  # adjust to your latest epoch folder
model, tokenizer = load_checkpoint(model_path, device)
results = evaluate_model(model, tokenizer,datasets["test"], device, k_values=[5,10])

print("Evaluation Results:", results)
