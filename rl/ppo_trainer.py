from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.extract_code import extract_code
from utils.prompts import create_prompt
from utils.pylint import analyze_code_with_pylint
from utils.rewards import extract_pylint_score
import torch

# def prepare_dataset(dataset_split):
#     prompts = [create_prompt(item["text"]) for item in dataset_split]
#     return Dataset.from_dict({"prompt": prompts})

class DummyRewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.tensor([0.0])  # dummy zero reward
    
def create_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    return model, tokenizer

def run_ppo_training(model_path, dataset_split, output_dir="./ppo_checkpoints"):
    # Load model/tokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = create_model_and_tokenizer(model_path, device)

    # Create reference model (frozen copy of model)
    ref_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

    # Create PPO Config
    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        save_steps=50,
        output_dir=output_dir
    )


    dummy_reward_model = DummyRewardModel()
    # Create PPOTrainer (no external reward model, using custom reward)
    ppo_trainer = PPOTrainer(
        args=config,
        model=model,
        ref_model=ref_model,
        reward_model=dummy_reward_model,  # We use our own reward
        value_model=model,   # Uses model by default
        train_dataset=dataset_split,
        processing_class=None,  # Uses default prompt processing
    )

    # ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

    # PPO Config
    # config = PPOConfig(
    #     learning_rate=1e-5,
    #     batch_size=2,
    #     mini_batch_size=1,
    #     save_steps=50,
    #     output_dir=output_dir
    # )

    # # Load model/tokenizer
    # model, tokenizer = create_model_and_tokenizer(model_path)

    # # # Prepare prompts
    # # prompts_ds = prepare_dataset(dataset_split)

    # # PPO Trainer
    # ppo_trainer = PPOTrainer(config, model, tokenizer)
    i=0
    for item in dataset_split:
        prompt = create_prompt(item['text'])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        response = ppo_trainer.policy_model.generate(**inputs, max_new_tokens=512)
        decoded_output = tokenizer.decode(response[0], skip_special_tokens=True)
        # test = analyze_code_with_pylint(respo)
        # Reward = static analysis of decoded_output
        # decoded_output = extract_code(decoded_output)
        print("\n\n", decoded_output, "\n")
        pylint_result = analyze_code_with_pylint(decoded_output)
        print(pylint_result)
        reward = extract_pylint_score(pylint_result)

        # PPO expects lists
        # ppo_trainer.step([prompt], [decoded_output], [reward])
        ppo_trainer.train()


        print(f"[{i}] Reward: {reward:.3f}")

        if i > 20:  # Run only on a few items for now (change as needed)
            break
        i+=1

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
