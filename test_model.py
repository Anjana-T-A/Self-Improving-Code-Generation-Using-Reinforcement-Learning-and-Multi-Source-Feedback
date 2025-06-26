import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from rl.evaluate_model import evaluate_model
from rl.model_loader import load_checkpoint
from data.datasets import load_datasets


datasets = load_datasets()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_path = "./ppo_combined2/epoch_3"
# model, tokenizer = load_checkpoint(model_path, device)
# results = evaluate_model(model, tokenizer, datasets["test"], device, "pyscore.py", "test.py", k_values=[5,10])


model_path = "./ppo_checkpoints/epoch_2"  # adjust to your latest epoch folder
model, tokenizer = load_checkpoint(model_path, device)
results = evaluate_model(model, tokenizer, datasets["test"], device, "pys1core.py", "t1est.py", k_values=[5,10])




# model_path = "./ppo_unit_test1/epoch_12"  # adjust to your latest epoch folder
print("Evaluation Results:", results)
