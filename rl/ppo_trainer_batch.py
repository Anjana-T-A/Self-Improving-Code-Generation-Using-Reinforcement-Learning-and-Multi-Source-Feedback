from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import load_dataset
from utils.extract_code import extract_code
from utils.prompts import create_prompt
from utils.pylint import analyze_code_with_pylint
from utils.rewards import extract_pylint_score
import torch
from torch.utils.data import DataLoader
import os
from torch.nn.utils.rnn import pad_sequence
from typing import List

def create_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side="left")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def collate_batch(batch, tokenizer, device):
    prompts = [create_prompt(item["text"], item["code"]) for item in batch]
    # Tokenize prompts with padding
    tokenizer.padding_side = "left"  # Ensure padding is on the left
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    return input_ids, attention_mask, prompts, batch


def run_ppo_epoch_training(
    model_path,
    output_dir="./ppo_checkpoints",
    batch_size=2,
    num_epochs=3,  # Added epoch control
    eval_every=1   # Evaluate every N epochs
):
    torch.cuda.empty_cache()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_path, device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)
    ref_model.eval()  # Freeze reference model

    # Load and split dataset
    dataset = load_dataset("mbpp", split="train")
    dataset = dataset.filter(lambda x: x['code'] is not None and len(x['code']) > 0)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']  # For evaluation

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, device)
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, device)
    )

    # PPO Config
    config = PPOConfig(
        learning_rate=1e-7,  # Reduced learning rate significantly
        batch_size=batch_size,
        mini_batch_size=batch_size,
        ppo_epochs=4,  # Inner PPO epochs per batch
        log_with=None,
        max_grad_norm=0.5,
        kl_penalty="kl",  # Explicitly set KL penalty
        init_kl_coef=0.03, # Initial KL penalty coefficient
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
            responses = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                do_sample=True,
                # top_p=0.9,
                # top_k=40,
                temperature=0.7,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask
            )

            query_tensors = []
            response_tensors = []
            reward_tensors = []

            for i in range(len(prompts)):
                prompt_len = input_ids[i].shape[-1]
                gen_tokens = responses[i][prompt_len:]
                decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                code_snippet = extract_code(decoded_output)
                pylint_result = analyze_code_with_pylint(code_snippet)
                reward = extract_pylint_score(pylint_result)

                query_tensors.append(input_ids[i])
                response_tensors.append(gen_tokens)
                reward_tensors.append(torch.tensor(reward, dtype=torch.float32).to(device))
                epoch_rewards.append(reward)
                print(f"\n=== Prompt ===\n{prompts[i]}")
                print(f"=== Generated Code ===\n{decoded_output}")
                print(f"=== Pylint Reward === {reward:.3f}")
                print("--"*100)

            # PPO Step
            ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

            # Print batch progress
            if (batch_idx + 1) % 5 == 0:
                avg_reward = sum(epoch_rewards[-5:])/5
                print(f"Epoch {epoch+1} | Batch {batch_idx+1} | Avg Reward: {avg_reward:.3f}")

        # End of epoch processing
        avg_epoch_reward = sum(epoch_rewards)/len(epoch_rewards)
        print(f"\n🏆 Epoch {epoch+1} Complete | Avg Reward: {avg_epoch_reward:.3f}")

        # Evaluation
        if (epoch + 1) % eval_every == 0:
            evaluate_model(model, tokenizer, test_dataloader, device)

        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")


def evaluate_model(model, tokenizer, dataloader, device):
    model.eval()
    eval_rewards = []

    print("\n===== Evaluating Model =====")

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, prompts, raw_batch) in enumerate(dataloader):

            responses = model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=False,  # Disable sampling for evaluation
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask
            )

            for i in range(len(prompts)):
                prompt_len = input_ids[i].shape[-1]
                gen_tokens = responses[i][prompt_len:]
                decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                code_snippet = extract_code(decoded_output)
                pylint_result = analyze_code_with_pylint(code_snippet)
                reward = extract_pylint_score(pylint_result)
                eval_rewards.append(reward)

    avg_eval_reward = sum(eval_rewards)/len(eval_rewards)
    print(f"📊 Evaluation Average Reward: {avg_eval_reward:.3f}")
    return avg_eval_reward







# from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
# from transformers import AutoTokenizer
# from datasets import load_dataset
# from utils.extract_code import extract_code
# from utils.prompts import create_prompt
# from utils.pylint import analyze_code_with_pylint
# from utils.rewards import extract_pylint_score
# import torch
# from torch.utils.data import DataLoader
# import os
# from torch.nn.utils.rnn import pad_sequence

# def create_model_and_tokenizer(model_path, device):
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side="left")
#     model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
#     model.to(device)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     return model, tokenizer


# def collate_batch(batch):
#     prompts = [create_prompt(item["text"], item["test_list"]) for item in batch]
#     return prompts, batch



# def run_ppo_epoch_training(
#     model_path, 
#     output_dir="./ppo_checkpoints", 
#     batch_size=2, 
#     num_epochs=3,  # Added epoch control
#     eval_every=1   # Evaluate every N epochs
# ):  
#     torch.cuda.empty_cache()
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
#     # Load model and tokenizer
#     model, tokenizer = create_model_and_tokenizer(model_path, device)
#     ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)
#     ref_model.eval()  # Freeze reference model
    
#     # Load and split dataset
#     dataset = load_dataset("mbpp", split="train")
#     dataset = dataset.filter(lambda x: x['code'] is not None and len(x['code']) > 0)
#     dataset = dataset.train_test_split(test_size=0.2, seed=42)
#     train_dataset = dataset['train']
#     test_dataset = dataset['test']  # For evaluation
    
#     # Create DataLoaders
#     train_dataloader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         collate_fn=collate_batch
#     )
    
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         collate_fn=collate_batch
#     )
    
#     # PPO Config
#     config = PPOConfig(
#         learning_rate=5e-6,
#         batch_size=batch_size,
#         mini_batch_size=batch_size,
#         ppo_epochs=4,  # Inner PPO epochs per batch
#         log_with=None,

#         max_grad_norm=0.5
#     )
    
#     ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
#     ppo_trainer.current_device = device
#     # Training Loop - Now epoch-based
#     for epoch in range(num_epochs):
#         epoch_rewards = []
#         model.train()
#         model.to(device)
#         ref_model.to(device)
#         torch.cuda.set_device(device)
#         print(f"\n======= Starting Epoch {epoch+1}/{num_epochs} =======")
        
#         for batch_idx, (prompts, raw_batch) in enumerate(train_dataloader):
#             inputs = tokenizer(prompts, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
#             attention_mask = inputs["attention_mask"].to(device)
#             tokenizer.padding_side = "left"

#             # Generate responses
#             responses = ppo_trainer.model.generate(
#                 input_ids=inputs["input_ids"],
#                 max_new_tokens=512,
#                 do_sample=True,
#                 # top_p=0.9,
#                 # top_k=40,
#                 temperature=0.7,
#                 repetition_penalty=1.2,
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#                 attention_mask=attention_mask
#             )
#             print(f"Currently selected device: {device}")
#             print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
#             print(f"torch.cuda.get_device_name(torch.cuda.current_device()): {torch.cuda.get_device_name(torch.cuda.current_device())}")

#             query_tensors = []
#             response_tensors = []
#             reward_tensors = []
            
#             for i in range(len(prompts)):
#                 prompt_len = inputs["input_ids"][i].shape[-1]
#                 gen_tokens = responses[i][prompt_len:]
#                 decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
#                 code_snippet = extract_code(decoded_output)
#                 pylint_result = analyze_code_with_pylint(code_snippet)
#                 reward = extract_pylint_score(pylint_result)
                
#                 query_tensors.append(inputs["input_ids"][i])
#                 response_tensors.append(gen_tokens)
#                 reward_tensors.append(torch.tensor(reward, dtype=torch.float32))
#                 epoch_rewards.append(reward)
#                 print(f"\n=== Prompt ===\n{prompts[i]}")
#                 print(f"=== Generated Code ===\n{decoded_output}")
#                 print(f"=== Pylint Reward === {reward:.3f}")
#                 print("--"*100)
#                 print(f"Currently selected device: {device}")
#                 print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
#                 print(f"torch.cuda.get_device_name(torch.cuda.current_device()): {torch.cuda.get_device_name(torch.cuda.current_device())}")
                
#                 # Pad sequences
#             # query_tensors = pad_sequence(query_tensors, batch_first=True).to(device)
#             # response_tensors = pad_sequence(response_tensors, batch_first=True).to(device)
#             # reward_tensors = torch.stack(reward_tensors).to(device)
            
#             # PPO Step
#             ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            
#             # Print batch progress
#             if (batch_idx + 1) % 5 == 0:
#                 avg_reward = sum(epoch_rewards[-5:])/5
#                 print(f"Epoch {epoch+1} | Batch {batch_idx+1} | Avg Reward: {avg_reward:.3f}")
        
#         # End of epoch processing
#         avg_epoch_reward = sum(epoch_rewards)/len(epoch_rewards)
#         print(f"\n🏆 Epoch {epoch+1} Complete | Avg Reward: {avg_epoch_reward:.3f}")
        
#         # Evaluation
#         if (epoch + 1) % eval_every == 0:
#             evaluate_model(model, tokenizer, test_dataloader, device)
        
#         # Save checkpoint
#         checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
#         model.save_pretrained(checkpoint_dir)
#         tokenizer.save_pretrained(checkpoint_dir)
#         print(f"Saved checkpoint to {checkpoint_dir}")









# def evaluate_model(model, tokenizer, dataloader, device):
#     model.eval()
#     eval_rewards = []
    
#     print("\n===== Evaluating Model =====")
    
#     with torch.no_grad():
#         for batch_idx, (prompts, raw_batch) in enumerate(dataloader):
#             inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            
#             responses = model.generate(
#                 input_ids=inputs["input_ids"],
#                 max_new_tokens=512,
#                 do_sample=False,  # Disable sampling for evaluation
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#             )
            
#             for i in range(len(prompts)):
#                 prompt_len = inputs["input_ids"][i].shape[-1]
#                 gen_tokens = responses[i][prompt_len:]
#                 decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
#                 code_snippet = extract_code(decoded_output)
#                 pylint_result = analyze_code_with_pylint(code_snippet)
#                 reward = extract_pylint_score(pylint_result)
#                 eval_rewards.append(reward)
    
#     avg_eval_reward = sum(eval_rewards)/len(eval_rewards)
#     print(f"📊 Evaluation Average Reward: {avg_eval_reward:.3f}")
#     return avg_eval_reward