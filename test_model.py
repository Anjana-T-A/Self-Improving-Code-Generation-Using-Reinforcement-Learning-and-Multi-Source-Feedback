import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from rl.evaluate_model import evaluate_model
from rl.model_loader import load_checkpoint
from data.datasets import load_datasets


# datasets = load_datasets()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_path = "./p1/epoch_3"
# model, tokenizer = load_checkpoint(model_path, device)
# results = evaluate_model(model, tokenizer, datasets["test"], device, "pyscore.py", "test.py", k_values=[5,10])


# model_path = "./ppo_checkpoints1/epoch_2"  # adjust to your latest epoch folder
# model, tokenizer = load_checkpoint(model_path, device)
# results = evaluate_model(model, tokenizer, datasets["test"], device, "pys1core.py", "t1est.py", k_values=[5,10])




# model_path = "./ppo_unit_test/epoch_2"  # adjust to your latest epoch folder
# model, tokenizer = load_checkpoint(model_path, device)
# results = evaluate_model(model, tokenizer, datasets["test"], device, "pyscore.py", "test.py", k_values=[5,10])







# result_file = "results1_batch4_e1.txt"

# with open(result_file, "a") as f:
#     for i in range(1, 12):  # p1 to p11
#         model_path = f"./p{i}/epoch_1"
#         try:
#             model, tokenizer = load_checkpoint(model_path, device)
#             results = evaluate_model(
#                 model,
#                 tokenizer,
#                 datasets["test"],
#                 device,
#                 "pyscore.py",
#                 "test.py",
#                 k_values=[5, 10]
#             )
#             f.write(f"Results for p{i}:\n")
#             f.write(str(results) + "\n\n")
#             print(f"[✓] Evaluation complete for p{i}")
#         except Exception as e:
#             f.write(f"Results for p{i}: ERROR - {str(e)}\n\n")
#             print(f"[!] Error in p{i}: {e}")

# result_file = "results_batch4_e2.txt"

# with open(result_file, "a") as f:
#     for i in range(6, 12):  # p1 to p11
#         model_path = f"./p{i}/epoch_2"
#         try:
#             model, tokenizer = load_checkpoint(model_path, device)
#             results = evaluate_model(
#                 model,
#                 tokenizer,
#                 datasets["test"],
#                 device,
#                 "pyscore.py",
#                 "test.py",
#                 k_values=[5, 10]
#             )
#             f.write(f"Results for p{i}:\n")
#             f.write(str(results) + "\n\n")
#             print(f"[✓] Evaluation complete for p{i}")
#         except Exception as e:
#             f.write(f"Results for p{i}: ERROR - {str(e)}\n\n")
#             print(f"[!] Error in p{i}: {e}")

from datasets import load_dataset

# Load and split dataset
dataset = load_dataset("mbpp", split="train")
dataset = dataset.filter(lambda x: x['code'] is not None and len(x['code']) > 0)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset['test']
print(f"[DEBUG] Loaded test dataset with {len(test_dataset)} examples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_file = "results_batch4_e2_check.txt"


with open(result_file, "a") as f:
    for i in range(4, 5):  # p5 to p11
        model_path = f"./p{i}/epoch_2"
        try:
            print(f"[DEBUG] Loading model from: {model_path}")
            model, tokenizer = load_checkpoint(model_path, device)

            print(f"[DEBUG] Evaluating model ppo{i}")
            results = evaluate_model(
                model,
                tokenizer,
                test_dataset,
                device,
                "pyscore.py",
                "test.py",
                k_values=[5, 10]
            )

            f.write(f"Results for p{i}:\n")
            f.write(str(results) + "\n\n")
            f.flush()
            print(f"[✓] Evaluation complete for p{i} epoch_2")

        except Exception as e:
            error_msg = f"Results for ppo{i}: ERROR - {str(e)}\n\n"
            f.write(error_msg)
            f.flush()
            print(f"[!] Error in ppo{i}: {e}")


# result_file = "results1_batch2_e1.txt"


# with open(result_file, "a") as f:
#     for i in range(1, 5):  # p5 to p11
#         model_path = f"./ppo{i}/epoch_1"
#         try:
#             print(f"[DEBUG] Loading model from: {model_path}")
#             model, tokenizer = load_checkpoint(model_path, device)

#             print(f"[DEBUG] Evaluating model p{i}")
#             results = evaluate_model(
#                 model,
#                 tokenizer,
#                 datasets["test"],
#                 device,
#                 "pyscore.py",
#                 "test.py",
#                 k_values=[5, 10]
#             )

#             f.write(f"Results for ppo{i}:\n epoch_1")
#             f.write(str(results) + "\n\n")
#             f.flush()
#             print(f"[✓] Evaluation complete for p{i}")

#         except Exception as e:
#             error_msg = f"Results for ppo{i}: ERROR - {str(e)}\n\n"
#             f.write(error_msg)
#             f.flush()
#             print(f"[!] Error in ppo{i}: {e}")
