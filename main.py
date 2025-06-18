from data.datasets import load_datasets
from rl.ppo_trainer_batch import run_ppo_epoch_training

datasets = load_datasets()
model_checkpoint = "ppo_checkpoints"

# 
# run_ppo_training("codellama/CodeLlama-7b-Python-hf", datasets["train"])
# run_ppo_training("Salesforce/codegen-2B-multi", datasets["train"],100)
# run_ppo_training("bigcode/starcoder2-3b", datasets["train"],100)
# run_ppo_batch_training("bigcode/starcoder2-3b", datasets["train"],100)
run_ppo_epoch_training("bigcode/starcoder2-3b",output_dir="./ppo_checkpoints", 
    batch_size=2, 
    num_epochs=3,  # Added epoch control
    eval_every=1  )
# run_ppo_epoch_training("codellama/CodeLlama-7b-Python-hf",output_dir="./ppo_checkpoints", 
#     batch_size=2, 
#     num_epochs=3,  # Added epoch control
#     eval_every=1  )
# evaluate_model(model_checkpoint, datasets["test"], k=10, unit_test_fn=None)

