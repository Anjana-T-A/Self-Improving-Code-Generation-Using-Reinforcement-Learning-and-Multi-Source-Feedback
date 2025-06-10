from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from rl.reward_model import compute_combined_reward
from data.datasets import load_datasets
from utils.prompts import create_prompt

def train_model_with_ppo(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    config = PPOConfig(model_name=model_name_or_path)
    ppo_trainer = PPOTrainer(config, model, tokenizer)

    datasets = load_datasets()
    train_set = datasets['train']

    for sample in train_set:
        prompt = create_prompt(sample['text'])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        generation_output = model.generate(input_ids, max_length=512)
        generated_code = tokenizer.decode(generation_output[0], skip_special_tokens=True)

        reward = compute_combined_reward(generated_code, sample)
        ppo_trainer.step([input_ids[0]], [generated_code], [reward])