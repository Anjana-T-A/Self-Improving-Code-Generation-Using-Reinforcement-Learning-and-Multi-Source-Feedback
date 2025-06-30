import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from data.datasets import load_datasets
from rl.ppo_trainer_batch import  run_ppo_epoch_training_unit_test
datasets = load_datasets()
model_checkpoint = "ppo_checkpoints"



run_ppo_epoch_training_unit_test("deepseek-ai/deepseek-coder-1.3b-instruct",output_dir="./ppo_unit_test", 
    batch_size=4, 
    num_epochs=4  # Added epoch control
      )

