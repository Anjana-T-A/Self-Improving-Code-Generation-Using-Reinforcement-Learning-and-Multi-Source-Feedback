from utils.extract_code import extract_code
from utils.pylint import analyze_code_with_pylint
from utils.rewards import extract_unit_test_score


def run_ppo_epoch_training_combined(
    model_path,
    output_dir="./ppo_combined",
    batch_size=2,
    num_epochs=3,
    weight_static=0.3,
    weight_unit_test=0.7,
):
    import os
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from trl import PPOTrainer, PPOConfig
    from transformers import AutoModelForCausalLMWithValueHead

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_path, device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)
    ref_model.eval()

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

    # Reward memory (baseline tracking)
    prompt_reward_baseline = {}

    for epoch in range(num_epochs):
        epoch_rewards = []
        model.train().to(device)
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

            query_tensors = []
            response_tensors = []
            reward_tensors = []

            for i in range(len(prompts)):
                prompt_len = input_ids[i].shape[-1]
                gen_tokens = responses[i][prompt_len:]
                decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                code_snippet = extract_code(decoded_output, "combined_generated.py")

                # Static analysis and unit test
                pylint_result = analyze_code_with_pylint(code_snippet, "combined_generated.py")
                static_reward = extract_pylint_score(pylint_result)

                unit_result = run_unit_tests(code_snippet, raw_batch[i]["test_list"], "combined_generated.py")
                unit_test_reward = extract_unit_test_score(unit_result)

                total_reward = weight_static * static_reward + weight_unit_test * unit_test_reward
                epoch_rewards.append(total_reward)

                # Baseline filtering logic
                prompt_text = prompts[i]
                baseline_reward = prompt_reward_baseline.get(prompt_text, float('-inf'))

                if total_reward > baseline_reward + 0.1:  # Optional threshold margin
                    query_tensors.append(input_ids[i])
                    response_tensors.append(gen_tokens)
                    reward_tensors.append(torch.tensor(total_reward, dtype=torch.float32).to(device))
                    prompt_reward_baseline[prompt_text] = total_reward
                else:
                    print(f"Skipping sample {i}: reward {total_reward:.2f} <= baseline {baseline_reward:.2f}")

                # Debug (optional)
                print("Prompt:", tokenizer.decode(input_ids[i], skip_special_tokens=True))
                print("Generated:", tokenizer.decode(responses[i], skip_special_tokens=True))
                print("Combined Reward:", total_reward)
                print("--" * 60)

            if len(query_tensors) == 0:
                print(f"Skipping batch {batch_idx+1}: no improved samples found.")
                continue

            # PPO Step
            ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}")

        # Save model
        checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")


def run_ppo_epoch_training_unit_test(
    model_path,
    output_dir="./ppo_unit_test",
    batch_size=2,
    num_epochs=3,
):
    import os
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from trl import PPOTrainer, PPOConfig
    from transformers import AutoModelForCausalLMWithValueHead

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_path, device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)
    ref_model.eval()

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
        learning_rate=5e-7,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        ppo_epochs=4,
        log_with=None,
        max_grad_norm=0.5,
        kl_penalty="kl",
        init_kl_coef=0.005,
        adap_kl_ctrl=True,
        target=6,
    )

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    ppo_trainer.current_device = device

    # Reward memory (baseline tracking)
    prompt_reward_baseline = {}

    # Training Loop
    for epoch in range(num_epochs):
        epoch_rewards = []
        model.train().to(device)
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

            query_tensors = []
            response_tensors = []
            reward_tensors = []

            for i in range(len(prompts)):
                prompt_len = input_ids[i].shape[-1]
                gen_tokens = responses[i][prompt_len:]
                decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                code_snippet = extract_code(decoded_output, "test_generated.py")
                unit_result = analyze_code_with_pylint(code_snippet, "test_generated.py")
                reward = extract_unit_test_score(unit_result)
                epoch_rewards.append(reward)

                # Baseline filtering logic
                prompt_text = prompts[i]
                baseline_reward = prompt_reward_baseline.get(prompt_text, float('-inf'))

                if reward > baseline_reward + 0.1:  # Optional threshold margin
                    query_tensors.append(input_ids[i])
                    response_tensors.append(gen_tokens)
                    reward_tensors.append(torch.tensor(reward, dtype=torch.float32).to(device))
                    prompt_reward_baseline[prompt_text] = reward
                else:
                    print(f"Skipping sample {i}: reward {reward:.2f} <= baseline {baseline_reward:.2f}")

            if len(query_tensors) == 0:
                print(f"Skipping batch {batch_idx+1}: no improved samples found.")
                continue

            # PPO Step
            ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

            # Batch log
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1} ")

        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
