import os

import torch

from rl.evaluate_model import evaluate_single_prompt
from rl.model_loader import load_checkpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from data.datasets import load_datasets
from rl.ppo_trainer_batch import  run_ppo_epoch_training_unit_test
datasets = load_datasets()
model_checkpoint = "ppo_checkpoints"



# run_ppo_epoch_training_unit_test("deepseek-ai/deepseek-coder-1.3b-instruct",output_dir="./ppo_unit_test", 
#     batch_size=4, 
#     num_epochs=4  # Added epoch control
#       )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_file = "qualitative.txt"

prompt = ''' Problem: 

            Write a Python function called `get_Pairs_Count` that solves the following problem:

            Write a python function to count the number of pairs whose sum is equal to ‘sum’.
            DO NOT repeat the prompt in response
            The function should have the following signature:

            def get_Pairs_Count(arr,n,sum):

            No comments needed in response and only must provide the whole code in response.'''
with open(result_file, "a") as f:
    for i in range(1, 12):  # p5 to p11
        model_path = f"./p{i}/epoch_2"
        try:
            print(f"[DEBUG] Loading model from: {model_path}")
            model, tokenizer = load_checkpoint(model_path, device)

            print(f"[DEBUG] Evaluating model ppo{i}")
            results = evaluate_single_prompt(
                model,
                tokenizer,
                prompt,[
"assert get_Pairs_Count([1,1,1,1],4,2) == 6",
"assert get_Pairs_Count([1,5,7,-1,5],5,6) == 3",
"assert get_Pairs_Count([1,-2,3],3,1) == 1"
],
                device,
                "b.py",
                "a.py"
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

