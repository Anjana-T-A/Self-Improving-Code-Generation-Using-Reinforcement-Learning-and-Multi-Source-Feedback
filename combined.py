
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo2",
    batch_size=2,
    num_epochs=2,
    weight_static=0.1,  # weight for pylint
    weight_unit_test=0.9,  # weight for unit tests
)
run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo3",
    batch_size=2,
    num_epochs=2,
    weight_static=0.2,  # weight for pylint
    weight_unit_test=0.8,  # weight for unit tests
)
run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo4",
    batch_size=2,
    num_epochs=2,
    weight_static=0.3,  # weight for pylint
    weight_unit_test=0.7,  # weight for unit tests
)
run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo5",
    batch_size=2,
    num_epochs=2,
    weight_static=0.4,  # weight for pylint
    weight_unit_test=0.6,  # weight for unit tests
)
run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo6",
    batch_size=2,
    num_epochs=2,
    weight_static=0.5,  # weight for pylint
    weight_unit_test=0.5,  # weight for unit tests
)
run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo7",
    batch_size=2,
    num_epochs=2,
    weight_static=0.6,  # weight for pylint
    weight_unit_test=0.4,  # weight for unit tests
)
run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo8",
    batch_size=2,
    num_epochs=2,
    weight_static=0.7,  # weight for pylint
    weight_unit_test=0.3,  # weight for unit tests
)
run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo9",
    batch_size=2,
    num_epochs=2,
    weight_static=0.8,  # weight for pylint
    weight_unit_test=0.2,  # weight for unit tests
)
run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo10",
    batch_size=2,
    num_epochs=2,
    weight_static=0.9,  # weight for pylint
    weight_unit_test=0.1,  # weight for unit tests
)
run_ppo_epoch_training_combined(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    output_dir="./ppo11",
    batch_size=2,
    num_epochs=2,
    weight_static=1,  # weight for pylint
    weight_unit_test=0,  # weight for unit tests
)
