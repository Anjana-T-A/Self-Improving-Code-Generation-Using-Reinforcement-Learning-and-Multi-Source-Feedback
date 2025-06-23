from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import load_dataset
from data.datasets import collate_batch
from utils.extract_code import extract_code
from utils.pylint import analyze_code_with_pylint
from utils.rewards import extract_pylint_score, extract_unit_test_score
import torch
from torch.utils.data import DataLoader
import os

def create_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer




def run_ppo_epoch_training_static(
    model_path,
    output_dir="./ppo_checkpoints",
    batch_size=2,
    num_epochs=3,  # Added epoch control
):
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_path, device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)
    ref_model.eval()  # Freeze reference model

    # Load and split dataset
    dataset = load_dataset("mbpp", split="train")
    dataset = dataset.filter(lambda x: x['code'] is not None and len(x['code']) > 0)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, device),
        drop_last=True,
        num_workers=0

    )

    # PPO Config
    config = PPOConfig(
        learning_rate=1e-5,  # Reduced learning rate significantly
        batch_size=batch_size,
        mini_batch_size=batch_size,
        ppo_epochs=4,  # Inner PPO epochs per batch
        log_with=None,
        max_grad_norm=0.2,
        kl_penalty="kl",  # Explicitly set KL penalty
        init_kl_coef=0.005, # Initial KL penalty coefficient
        adap_kl_ctrl=True, # Use adaptive KL control
        target=.02,           # Target KL value

    )

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    ppo_trainer.current_device = device
    # Training Loop - Now epoch-based
    for epoch in range(num_epochs):
        epoch_rewards = []
        model.train()
        model.to(device)
        ref_model.to(device)
        torch.cuda.set_device(device)
        print(f"\n======= Starting Epoch {epoch+1}/{num_epochs} =======")

        for batch_idx, (input_ids, attention_mask, prompts, raw_batch) in enumerate(train_dataloader):
            # Generate responses
            responses = ppo_trainer.model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                top_k=40,
                temperature=0.7,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask
            )
            # print(f"Currently selected device: {device}")
            # print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
            # print(f"torch.cuda.get_device_name(torch.cuda.current_device()): {torch.cuda.get_device_name(torch.cuda.current_device())}")

            query_tensors = []
            response_tensors = []
            reward_tensors = []
            
            for i in range(len(prompts)):
                prompt_len = input_ids[i].shape[-1]
                gen_tokens = responses[i][prompt_len:]
                decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                code_snippet = extract_code(decoded_output,"generated.py")
                pylint_result = analyze_code_with_pylint(code_snippet,"generated.py")
                reward = extract_pylint_score(pylint_result)
                query_tensors.append(input_ids[i])
                response_tensors.append(gen_tokens)
                reward_tensors.append(torch.tensor(reward, dtype=torch.float64).to(device))
                epoch_rewards.append(reward)
                # print(f"\n=== Prompt ===\n{prompts[i]}")
                # print(f"=== Generated Code ===\n{decoded_output}")
                # print(f"=== Pylint Reward === {reward:.3f}")
                # print("--"*100)
                # print(f"Currently selected device: {device}")
                # print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
                # print(f"torch.cuda.get_device_name(torch.cuda.current_device()): {torch.cuda.get_device_name(torch.cuda.current_device())}")
            if len(query_tensors) != config.batch_size:
                print(f"Skipping batch {batch_idx+1}: expected batch size {config.batch_size}, got {len(query_tensors)}")
                continue

            # PPO Step
            ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

            # Print batch progress
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1} ")

        checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")



def run_ppo_epoch_training_unit_test(
    model_path,
    output_dir="./ppo_unit_test",
    batch_size=2,
    num_epochs=3,  # Added epoch control
):
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_path, device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)
    ref_model.eval()  # Freeze reference model

    # Load and split dataset
    dataset = load_dataset("mbpp", split="train")
    dataset = dataset.filter(lambda x: x['code'] is not None and len(x['code']) > 0)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, device),
        drop_last=True,
        num_workers=0

    )

    # PPO Config
    config = PPOConfig(
        learning_rate=5e-7,  # Reduced learning rate significantly
        batch_size=batch_size,
        mini_batch_size=batch_size,
        ppo_epochs=4,  # Inner PPO epochs per batch
        log_with=None,
        max_grad_norm=0.5,
        kl_penalty="kl",  # Explicitly set KL penalty
        init_kl_coef=0.005, # Initial KL penalty coefficient
        adap_kl_ctrl=True, # Use adaptive KL control
        target=6,           # Target KL value

    )

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    ppo_trainer.current_device = device
    # Training Loop - Now epoch-based
    for epoch in range(num_epochs):
        epoch_rewards = []
        model.train()
        model.to(device)
        ref_model.to(device)
        torch.cuda.set_device(device)
        print(f"\n======= Starting Epoch {epoch+1}/{num_epochs} =======")

        for batch_idx, (input_ids, attention_mask, prompts, raw_batch) in enumerate(train_dataloader):
            # Generate responses
            responses = ppo_trainer.model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.7,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask
            )
            # print(f"Currently selected device: {device}")
            # print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
            # print(f"torch.cuda.get_device_name(torch.cuda.current_device()): {torch.cuda.get_device_name(torch.cuda.current_device())}")

            query_tensors = []
            response_tensors = []
            reward_tensors = []
            
            for i in range(len(prompts)):
                prompt_len = input_ids[i].shape[-1]
                gen_tokens = responses[i][prompt_len:]
                decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                code_snippet = extract_code(decoded_output,"test_generated.py")
                unit_result = analyze_code_with_pylint(code_snippet,"test_generated.py")
                reward = extract_unit_test_score(unit_result)
                query_tensors.append(input_ids[i])
                response_tensors.append(gen_tokens)
                reward_tensors.append(torch.tensor(reward, dtype=torch.float32).to(device))
                epoch_rewards.append(reward)
                # print(f"\n=== Prompt ===\n{prompts[i]}")
                # print(f"=== Generated Code ===\n{decoded_output}")
                # print(f"=== Pylint Reward === {reward:.3f}")
                # print("--"*100)
                # print(f"Currently selected device: {device}")
                # print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
                # print(f"torch.cuda.get_device_name(torch.cuda.current_device()): {torch.cuda.get_device_name(torch.cuda.current_device())}")
            if len(query_tensors) != config.batch_size:
                print(f"Skipping batch {batch_idx+1}: expected batch size {config.batch_size}, got {len(query_tensors)}")
                continue

            # PPO Step
            ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

            # Print batch progress
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1} ")

        checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
