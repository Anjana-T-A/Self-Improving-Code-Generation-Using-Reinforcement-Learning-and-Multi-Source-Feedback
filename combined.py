
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from data.datasets import load_datasets
from rl.ppo_trainer_batch import run_ppo_epoch_training_combined
datasets = load_datasets()



run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo_combined",
    batch_size=2,
    num_epochs=20,
    weight_static=0.4,  # weight for pylint
    weight_unit_test=0.6,  # weight for unit tests
)