import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from data.datasets import load_datasets
from rl.ppo_trainer import run_ppo_training
from rl.ppo_trainer_batch import run_ppo_epoch_training_static, run_ppo_epoch_training_unit_test
datasets = load_datasets()



run_ppo_epoch_training_static("deepseek-ai/deepseek-coder-1.3b-instruct",output_dir="./ppo_checkpoints", 
    batch_size=2, 
    num_epochs=20  # Added epoch control
      )