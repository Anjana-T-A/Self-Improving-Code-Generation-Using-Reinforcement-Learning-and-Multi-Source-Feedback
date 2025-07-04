
import os

import torch

from rl.evaluate_model import evaluate_model
from rl.model_loader import load_checkpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from data.datasets import load_datasets
from rl.ppo_trainer_batch import run_ppo_epoch_training_combined
datasets = load_datasets()



# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./ppo1",
#     batch_size=2,
#     num_epochs=2,
#     weight_static=0,  # weight for pylint
#     weight_unit_test=1,  # weight for unit tests
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./ppo2",
#     batch_size=2,
#     num_epochs=2,
#     weight_static=0.1,  # weight for pylint
#     weight_unit_test=0.9,  # weight for unit tests
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./ppo3",
#     batch_size=2,
#     num_epochs=2,
#     weight_static=0.2,  # weight for pylint
#     weight_unit_test=0.8,  # weight for unit tests
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./ppo4",
#     batch_size=2,
#     num_epochs=2,
#     weight_static=0.3,  # weight for pylint
#     weight_unit_test=0.7,  # weight for unit tests
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./ppo5",
#     batch_size=2,
#     num_epochs=2,
#     weight_static=0.4,  # weight for pylint
#     weight_unit_test=0.6,  # weight for unit tests
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./ppo6",
#     batch_size=2,
#     num_epochs=2,
#     weight_static=0.5,  # weight for pylint
#     weight_unit_test=0.5,  # weight for unit tests
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./ppo7",
#     batch_size=2,
#     num_epochs=2,
#     weight_static=0.6,  # weight for pylint
#     weight_unit_test=0.4,  # weight for unit tests
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./ppo8",
#     batch_size=2,
#     num_epochs=2,
#     weight_static=0.7,  # weight for pylint
#     weight_unit_test=0.3,  # weight for unit tests
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./ppo9",
#     batch_size=2,
#     num_epochs=2,
#     weight_static=0.8,  # weight for pylint
#     weight_unit_test=0.2,  # weight for unit tests
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./ppo10",
#     batch_size=2,
#     num_epochs=2,
#     weight_static=0.9,  # weight for pylint
#     weight_unit_test=0.1,  # weight for unit tests
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./ppo11",
#     batch_size=2,
#     num_epochs=2,
#     weight_static=1,  # weight for pylint
#     weight_unit_test=0,  # weight for unit tests
# )






# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./p1",
#     batch_size=4,
#     num_epochs=2,
#     weight_static=0.0,
#     weight_unit_test=1.0,
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./p2",
#     batch_size=4,
#     num_epochs=2,
#     weight_static=0.1,
#     weight_unit_test=0.9,
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./p3",
#     batch_size=4,
#     num_epochs=2,
#     weight_static=0.2,
#     weight_unit_test=0.8,
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./p4",
#     batch_size=4,
#     num_epochs=2,
#     weight_static=0.3,
#     weight_unit_test=0.7,
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./p5",
#     batch_size=4,
#     num_epochs=2,
#     weight_static=0.4,
#     weight_unit_test=0.6,
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./p6",
#     batch_size=4,
#     num_epochs=2,
#     weight_static=0.5,
#     weight_unit_test=0.5,
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./p7",
#     batch_size=4,
#     num_epochs=2,
#     weight_static=0.6,
#     weight_unit_test=0.4,
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./p8",
#     batch_size=4,
#     num_epochs=2,
#     weight_static=0.7,
#     weight_unit_test=0.3,
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./p9",
#     batch_size=4,
#     num_epochs=2,
#     weight_static=0.8,
#     weight_unit_test=0.2,
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./p10",
#     batch_size=4,
#     num_epochs=2,
#     weight_static=0.9,
#     weight_unit_test=0.1,
# )
# run_ppo_epoch_training_combined(
#     "deepseek-ai/deepseek-coder-1.3b-instruct",
#     output_dir="./p11",
#     batch_size=4,
#     num_epochs=2,
#     weight_static=1.0,
#     weight_unit_test=0.0,
# )




# # Assuming you already have these functions and variables available:
# # - load_checkpoint(model_path, device)
# # - evaluate_model(model, tokenizer, test_dataset, device, score_script, test_script)
# # - datasets["test"]
# # - device

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# result_file = "results.txt"

# with open(result_file, "w") as f:
#     for i in range(1, 12):  # p1 to p11
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_file = "results_batch2_e2.txt"


with open(result_file, "a") as f:
    for i in range(9, 12):  # p5 to p11
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
    for i in range(5, 12):  # p5 to p11
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
