import os
from rl.ppo_trainer_batch import create_model_and_tokenizer
from utils.prompts import create_prompt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl.evaluate_model import  evaluate_model
from data.datasets import load_datasets


datasets = load_datasets()
model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"  # Change for each model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
metrics = evaluate_model(model, tokenizer, datasets["test"], device, "becnc1hmark_train.py", "test_co1detrain.py", k_values=[5,10])
print(metrics)
