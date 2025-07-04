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

from utils.unit_tests import run_unit_tests

def create_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer




# def run_ppo_epoch_training_static(
#     model_path,
#     output_dir="./ppo_checkpoints",
#     batch_size=2,
#     num_epochs=3,  # Added epoch control
# ):
#     torch.cuda.empty_cache()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Load model and tokenizer
#     model, tokenizer = create_model_and_tokenizer(model_path, device)
#     ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)
#     ref_model.eval()  # Freeze reference model

#     # Load and split dataset
#     dataset = load_dataset("mbpp", split="train")
#     dataset = dataset.filter(lambda x: x['code'] is not None and len(x['code']) > 0)
#     dataset = dataset.train_test_split(test_size=0.2, seed=42)
#     train_dataset = dataset['train']

#     # Create DataLoaders
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=lambda batch: collate_batch(batch, tokenizer, device),
#         drop_last=True,
#         num_workers=0

#     )

#     # PPO Config
#     config = PPOConfig(
#         learning_rate=1e-6,  # Reduced learning rate significantly
#         batch_size=batch_size,
#         mini_batch_size=batch_size,
#         ppo_epochs=4,  # Inner PPO epochs per batch
#         log_with=None,
#         max_grad_norm=1.0,
#         kl_penalty="kl",  # Explicitly set KL penalty
#         init_kl_coef=0.05, # Initial KL penalty coefficient
#         adap_kl_ctrl=True, # Use adaptive KL control
#         target=.2,           # Target KL value

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

#         for batch_idx, (input_ids, attention_mask, prompts, raw_batch) in enumerate(train_dataloader):
#             # Generate responses
#             responses = ppo_trainer.model.generate(
#                 input_ids=input_ids,
#                 max_new_tokens=256,
#                 do_sample=True,
#                 top_p=0.9,
#                 top_k=40,
#                 temperature=0.75,
#                 repetition_penalty=1.2,
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#                 attention_mask=attention_mask
#             )
#             # print(f"Currently selected device: {device}")
#             # print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
#             # print(f"torch.cuda.get_device_name(torch.cuda.current_device()): {torch.cuda.get_device_name(torch.cuda.current_device())}")

#             query_tensors = []
#             response_tensors = []
#             reward_tensors = []
            
#             for i in range(len(prompts)):
#                 prompt_len = input_ids[i].shape[-1]
#                 gen_tokens = responses[i][prompt_len:]
#                 print(f"Generated tokens length: {len(gen_tokens)}, gen_tokens: {gen_tokens}")
#                 decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
#                 code_snippet = extract_code(decoded_output,"generated.py")
#                 pylint_result = analyze_code_with_pylint(code_snippet,"generated.py")
#                 reward = extract_pylint_score(pylint_result)
#                 query_tensors.append(input_ids[i])
#                 response_tensors.append(gen_tokens)
#                 reward_tensors.append(torch.tensor(reward, dtype=torch.float64).to(device))
#                 epoch_rewards.append(reward)
#                 print(input_ids[i])
#                 # print(f"\n=== Prompt ===\n{prompts[i]}")
#                 # print(f"=== Generated Code ===\n{decoded_output}")
#                 # print(f"=== Pylint Reward === {reward:.3f}")
#                 # print("--"*100)
#                 # print(f"Currently selected device: {device}")
#                 # print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
#                 # print(f"torch.cuda.get_device_name(torch.cuda.current_device()): {torch.cuda.get_device_name(torch.cuda.current_device())}")
#             if len(query_tensors) != config.batch_size:
#                 print(f"Skipping batch {batch_idx+1}: expected batch size {config.batch_size}, got {len(query_tensors)}")
#                 continue

#             # PPO Step
#             ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

#             # Print batch progress
#             if (batch_idx + 1) % 5 == 0:
#                 print(f"Epoch {epoch+1} | Batch {batch_idx+1} ")

#         checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
#         model.save_pretrained(checkpoint_dir)
#         tokenizer.save_pretrained(checkpoint_dir)
#         print(f"Saved checkpoint to {checkpoint_dir}")

def run_ppo_epoch_training_static(
    model_path,
    output_dir="./ppo_checkpoints",
    batch_size=2,
    num_epochs=3,
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
        learning_rate=4e-7,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        ppo_epochs=4,
        log_with=None,
        max_grad_norm=0.5,
        kl_penalty="kl",
        init_kl_coef=0.1,
        adap_kl_ctrl=True,
        target=0.2,
    )

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    ppo_trainer.current_device = device

    for epoch in range(num_epochs):
        epoch_rewards = []
        model.train()
        model.to(device)
        ref_model.to(device)
        torch.cuda.set_device(device)
        print(f"\n======= Starting Epoch {epoch+1}/{num_epochs} =======")

        for batch_idx, (input_ids, attention_mask, prompts, raw_batch) in enumerate(train_dataloader):
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
            # prompt_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            # print(prompt_text)
            query_tensors = []
            response_tensors = []
            reward_tensors = []

            for i in range(len(prompts)):
                prompt_len = input_ids[i].shape[-1]
                gen_tokens = responses[i][prompt_len:]
                decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                code_snippet = extract_code(decoded_output, "combined_generated.py")

                # Run static analysis
                pylint_result = analyze_code_with_pylint(code_snippet, "combined_generated.py")
                reward = extract_pylint_score(pylint_result)

                # Weighted combination
                query_tensors.append(input_ids[i])
                response_tensors.append(gen_tokens)
                reward_tensors.append(torch.tensor(reward, dtype=torch.float32).to(device))
                epoch_rewards.append(reward)
                                # ...existing code...
                # print("query_tensors:")
                # for t in query_tensors:
                #     print("  shape:", t.shape, "dtype:", t.dtype, "sample:", t[:10])
                # print("response_tensors:")
                # for t in response_tensors:
                #     print("  shape:", t.shape, "dtype:", t.dtype, "sample:", t[:10])
                # print("reward_tensors:")
                # for t in reward_tensors:
                #     print("  value:", t.item(), "dtype:", t.dtype)
                # PPO Step
                # ...existing code...
                # Debug (optional)
                # print(f"\n=== Prompt ===\n{prompts[i]}")
                # print(f"=== Generated Code ===\n{decoded_output}")
                # print(f"Static Reward: {static_reward:.3f} | Unit Test Reward: {unit_test_reward:.3f} | Combined: {total_reward:.3f}")
                # print("--" * 80)

                print("Prompt:", tokenizer.decode(input_ids[i], skip_special_tokens=True))
                print("Generated:", tokenizer.decode(responses[i], skip_special_tokens=True))
                print("gen_tokens:", gen_tokens)
                print("Decoded output:", decoded_output)
            if len(query_tensors) != config.batch_size:
                print(f"Skipping batch {batch_idx+1}: expected batch size {config.batch_size}, got {len(query_tensors)}")
                continue

            # PPO Step
            ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}")
            query_tensors.clear()
            response_tensors.clear()
            reward_tensors.clear()
        # Save model
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
        learning_rate=3e-7,  # Reduced learning rate significantly
        batch_size=batch_size,
        mini_batch_size=batch_size,
        ppo_epochs=4,  # Inner PPO epochs per batch
        log_with=None,
        max_grad_norm=0.5,
        kl_penalty="kl",  # Explicitly set KL penalty
        init_kl_coef=0.1, # Initial KL penalty coefficient
        adap_kl_ctrl=True, # Use adaptive KL control
        target=.1,           # Target KL value

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
                top_p=0.75,
                top_k=50,
                temperature=0.7,
                repetition_penalty=1.2,
                length_penalty=1.0,
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
                unit_result = run_unit_tests(code_snippet,raw_batch[i]["test_list"],"test_generated.py")
                reward = extract_unit_test_score(unit_result )
                print(f"Unit test result: {unit_result}"    )
                print(f"Reward: {reward}")
                query_tensors.append(input_ids[i])
                response_tensors.append(gen_tokens)
                reward_tensors.append(torch.tensor(reward, dtype=torch.float32).to(device))
                epoch_rewards.append(reward)
                # print(f"\n=== Prompt ===\n{prompts[i]}")
                # print(f"=== Generated Code ===\n{decoded_output}")
                # print(f"=== Unit test Reward === {reward:.3f}")
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



def run_ppo_epoch_training_combined(
    model_path,
    output_dir="./ppo_combined",
    batch_size=2,
    num_epochs=3,
    weight_static=0.3,  # weight for pylint
    weight_unit_test=0.7,  # weight for unit tests
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
        learning_rate=4e-7,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        ppo_epochs=4,
        log_with=None,
        max_grad_norm=0.5,
        kl_penalty="kl",
        init_kl_coef=0.1,
        adap_kl_ctrl=True,
        target=0.2,
    )

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    ppo_trainer.current_device = device

    for epoch in range(num_epochs):
        epoch_rewards = []
        model.train()
        model.to(device)
        ref_model.to(device)
        torch.cuda.set_device(device)
        print(f"\n======= Starting Epoch {epoch+1}/{num_epochs} =======")

        for batch_idx, (input_ids, attention_mask, prompts, raw_batch) in enumerate(train_dataloader):
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
            # prompt_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            # print(prompt_text)
            query_tensors = []
            response_tensors = []
            reward_tensors = []

            for i in range(len(prompts)):
                prompt_len = input_ids[i].shape[-1]
                gen_tokens = responses[i][prompt_len:]
                decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                code_snippet = extract_code(decoded_output, "combined_generated.py")

                # Run static analysis
                pylint_result = analyze_code_with_pylint(code_snippet, "combined_generated.py")
                static_reward = extract_pylint_score(pylint_result)

                # Run unit test
                unit_result = run_unit_tests(code_snippet, raw_batch[i]["test_list"], "combined_generated.py")
                unit_test_reward = extract_unit_test_score(unit_result)

                # Weighted combination
                total_reward = weight_static * static_reward + weight_unit_test * unit_test_reward
                query_tensors.append(input_ids[i])
                response_tensors.append(gen_tokens)
                reward_tensors.append(torch.tensor(total_reward, dtype=torch.float32).to(device))
                epoch_rewards.append(total_reward)
                                # ...existing code...
                # print("query_tensors:")
                # for t in query_tensors:
                #     print("  shape:", t.shape, "dtype:", t.dtype, "sample:", t[:10])
                # print("response_tensors:")
                # for t in response_tensors:
                #     print("  shape:", t.shape, "dtype:", t.dtype, "sample:", t[:10])
                # print("reward_tensors:")
                # for t in reward_tensors:
                #     print("  value:", t.item(), "dtype:", t.dtype)
                # PPO Step
                # ...existing code...
                # Debug (optional)
                # print(f"\n=== Prompt ===\n{prompts[i]}")
                # print(f"=== Generated Code ===\n{decoded_output}")
                # print(f"Static Reward: {static_reward:.3f} | Unit Test Reward: {unit_test_reward:.3f} | Combined: {total_reward:.3f}")
                # print("--" * 80)

                # print("Prompt:", tokenizer.decode(input_ids[i], skip_special_tokens=True))
                # print("Generated:", tokenizer.decode(responses[i], skip_special_tokens=True))
                # print("gen_tokens:", gen_tokens)
                # print("Decoded output:", decoded_output)
            if len(query_tensors) != config.batch_size:
                print(f"Skipping batch {batch_idx+1}: expected batch size {config.batch_size}, got {len(query_tensors)}")
                continue

            # PPO Step
            ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}")
            query_tensors.clear()
            response_tensors.clear()
            reward_tensors.clear()
        # Save model
        checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")





