
import os

import torch

from rl.evaluate_model import evaluate_model
from rl.model_loader import load_checkpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from data.datasets import load_datasets
from rl.ppo_trainer_batch import run_ppo_epoch_training_combined
datasets = load_datasets()



run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo1",
    batch_size=2,
    num_epochs=2,
    weight_static=0,  # weight for pylint
    weight_unit_test=1,  # weight for unit tests
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_file = "results_batch2_e2.txt"


with open(result_file, "a") as f:
    for i in range(1, 12): 
        model_path = f"./ppo{i}/epoch_2"
        try:
            print(f"[DEBUG] Loading model from: {model_path}")
            model, tokenizer = load_checkpoint(model_path, device)

            print(f"[DEBUG] Evaluating model ppo{i}")
            results = evaluate_model(
                model,
                tokenizer,
                datasets["test"],
                device,
                "pyscore.py",
                "test.py",
                k_values=[5, 10]
            )

            f.write(f"Results for ppo{i}:\n")
            f.write(str(results) + "\n\n")
            f.flush()
            print(f"[✓] Evaluation complete for p{i}")

        except Exception as e:
            error_msg = f"Results for ppo{i}: ERROR - {str(e)}\n\n"
            f.write(error_msg)
            f.flush()
            print(f"[!] Error in ppo{i}: {e}")


result_file = "results1_batch2_e1.txt"


with open(result_file, "a") as f:
    for i in range(1, 12):  # p5 to p11
        model_path = f"./ppo{i}/epoch_1"
        try:
            print(f"[DEBUG] Loading model from: {model_path}")
            model, tokenizer = load_checkpoint(model_path, device)

            print(f"[DEBUG] Evaluating model p{i}")
            results = evaluate_model(
                model,
                tokenizer,
                datasets["test"],
                device,
                "pyscore.py",
                "test.py",
                k_values=[5, 10]
            )

            f.write(f"Results for ppo{i}:\n")
            f.write(str(results) + "\n\n")
            f.flush()
            print(f"[✓] Evaluation complete for p{i}")

        except Exception as e:
            error_msg = f"Results for ppo{i}: ERROR - {str(e)}\n\n"
            f.write(error_msg)
            f.flush()
            print(f"[!] Error in ppo{i}: {e}")
