from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from utils.extract_code import extract_code
from utils.prompts import create_prompt
from utils.pylint import analyze_code_with_pylint
from utils.rewards import extract_pylint_score
import torch


def create_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def run_ppo_training(model_path, dataset_split, output_dir="./ppo_checkpoints"):
    torch.cuda.empty_cache()
    # Set device to GPU 1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # Load model/tokenizer and move model to device
    model, tokenizer = create_model_and_tokenizer(model_path, device)

    # Load ref model and move to device
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    # Force Accelerate to use cuda:1
    # PPO Config
    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=1,
        mini_batch_size=1
    )
    

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    # ppo_trainer.current_device = "cuda:1"
    ppo_trainer.current_device = device
    # Ensure the model is on the correct device
    model.to(device)
    ref_model.to(device)


    i = 0
    for item in dataset_split:
        prompt = create_prompt(item['text'], item['test_list'])
        
        # Tokenize and move inputs to device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate response on device
        response = ppo_trainer.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        

        generated_tokens = response[0][inputs["input_ids"].shape[-1]:]
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        code_snippet = extract_code(decoded_output)
        pylint_result = analyze_code_with_pylint(code_snippet)
        reward = extract_pylint_score(pylint_result)
        print(f"\n=== Prompt ===\n{prompt}")
        print(f"=== Generated Code ===\n{decoded_output}")
        print(f"=== Pylint Reward === {reward:.3f}")

        # Prepare PPO inputs: move tensors to device
        query_tensor = inputs["input_ids"].squeeze(0).to(device)  # Already on device but just in case
        response_tensor = response[0][query_tensor.shape[-1]:].to(device)  # Only new tokens on device

        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)  # Reward tensor on device

       # PPO step
        ppo_trainer.step([query_tensor], [response_tensor], [reward_tensor])

        print(f"[{i}] Reward: {reward:.3f}")

        if i ==2:
            break
        i += 1
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)