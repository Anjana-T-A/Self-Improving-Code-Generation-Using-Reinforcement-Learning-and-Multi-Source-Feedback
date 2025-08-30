import ast
import os
import torch
from data.datasets import get_dataloader
from rl.ppo_trainer_batch import collate_batch
from utils.extract_code import extract_code
from utils.pylint import analyze_code_with_pylint
from utils.rewards import extract_pylint_score
from utils.unit_tests import run_unit_tests
from torch.utils.data import DataLoader
from math import comb

def is_syntax_valid(code: str) -> bool:
    """Check if the code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def unbiased_pass_at_k(c, n, k):
    """Unbiased estimator for pass@k as per OpenAI Codex paper."""
    if c == 0:
        return 0.0
    if k > n:
        return 1.0
    return 1.0 - (comb(n - c, k) / comb(n, k))

def evaluate_model(model, tokenizer, test_dataset, device,filepath1, filepath2, k_values=[1, 5, 10]):
    total_samples = 0
    pylint_score_sum = 0.0
    pylint_score_count = 0
    syntax_valid_counts = 0
    pass_at_k_sums = {k: 0.0 for k in k_values}
    dataloader = get_dataloader(test_dataset, tokenizer, device)
    print("\n===== Evaluating Unbiased Pass@k and Compiler Error Rate =====")

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, prompts, raw_batch) in enumerate(dataloader):
            batch_size = len(prompts)
            total_samples += batch_size
            for i in range(batch_size):
                # print(i)
                try:
                    if i >= len(raw_batch) or i >= len(input_ids) or i >= len(attention_mask):
                        print(f"Skipping sample #{i} due to length mismatch — batch #{batch_idx}")
                        print(f"  Sizes — input_ids: {len(input_ids)}, attention_mask: {len(attention_mask)}, prompts: {len(prompts)}, raw_batch: {len(raw_batch)}")
                        continue
                    prompt_input_ids = input_ids[i].unsqueeze(0).to(device)
                    prompt_attention_mask = attention_mask[i].unsqueeze(0).to(device)
                    all_pass = []
                    all_syntax = []

                    for _ in range(max(k_values)):
                        try:
                            outputs = model.generate(
                                input_ids=prompt_input_ids,
                                attention_mask=prompt_attention_mask,
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.7,
                                top_k=50,
                                top_p=0.95,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id
                            )
                            # print(f"🔎 outputs shape: {outputs.shape}")
                            output_seq = outputs[0]  # outputs is shape [1, seq_len]
                            prompt_len = prompt_input_ids.shape[-1]
                            # print("check 4")
                            if prompt_len >= output_seq.shape[-1]:
                                print(f"⚠️ Skipping due to prompt_len ({prompt_len}) >= output length ({output_seq.shape[-1]})")
                                continue

                            gen_tokens = output_seq[prompt_len:]

                            decoded_code = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                            print(f" Prompt:\n{prompts[i]}")
                            code_snippet = extract_code(decoded_code, filepath1)
                            if not isinstance(code_snippet, str):
                                code_snippet = ""
                            print(f" Code Snippet:\n{code_snippet}")
                            syntax_ok = is_syntax_valid(code_snippet)
                            all_syntax.append(syntax_ok)
                           
                            if syntax_ok:
                                test_list = raw_batch[i]["test_list"]
                                if not test_list:  # Check if the list is empty
                                    print(f" Skipping unit tests for sample #{i} because test_list is empty.")
                                    passed = False  # Or handle it differently, e.g., set passed = True
                                else:
                                    test_result = run_unit_tests(code_snippet, test_list, filepath2)
                                    print(f"🔵 Unit Test Result: {test_result}")
                                    passed = test_result["passed"]
                            else:
                                passed = False
                            if passed==False: 
                                all_pass.append(passed)
                            else:
                                all_pass.append(True)
                            # --- Pylint score calculation ---
                            pylint_result = analyze_code_with_pylint(code_snippet, filepath1)
                            # print(f"🔵 Pylint Messages: {pylint_result}")
                            pylint_score_sum += pylint_result["score"]
                            pylint_score_count += 1

                        except Exception as e:
                            print(f"\n Exception at batch #{batch_idx}, sample #{i}")
                            print(f"  raw_batch size: {len(raw_batch)}, input_ids shape: {input_ids.shape}")
                            print(f"  Exception: {e}")
                            continue

                    syntax_valid_counts += sum(all_syntax)

                    c = sum(all_pass)  # number of correct completions
                    n = len(all_pass)  # number of completions generated
                     
                    for k in k_values:
                        pass_at_k_sums[k] += unbiased_pass_at_k(c, n, k)
                   
                except Exception as e:
                    print(f"\n Unexpected exception at batch #{batch_idx}, sample #{i}")
                    print(f"  Exception: {e}")
                    continue

    num_prompts = total_samples
    compiler_error_rate = 1 - (syntax_valid_counts / (num_prompts * max(k_values)))
    avg_pylint_score = pylint_score_sum / pylint_score_count if pylint_score_count > 0 else 0.0

    
    print(f"Average Pylint Score: {avg_pylint_score:.3f}")

    print(f"Total prompts evaluated: {num_prompts}")
    print(f"Compiler Error Rate: {compiler_error_rate:.3f}")

    for k in k_values:
        pass_at_k = pass_at_k_sums[k] / num_prompts
        print(f"Unbiased Pass@{k}: {pass_at_k:.3f}")

    return {
        "compiler_error_rate": compiler_error_rate,
        "avg_pylint_score": avg_pylint_score,
        **{f"pass@{k}": pass_at_k_sums[k] / num_prompts for k in k_values}
    }




def evaluate_single_prompt(model, tokenizer, prompt, test_list, device, filepath_temp_code, filepath_test_dir):
    model.eval()

    print(f"\n🔹 Prompt:\n{prompt}")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    output_seq = outputs[0]
    prompt_len = input_ids.shape[-1]
    gen_tokens = output_seq[prompt_len:]
    decoded_code = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    print(f"\n🟢 Generated Code:\n{decoded_code}")

    # Extract and validate code (assume extract_code & is_syntax_valid exist)
    code_snippet = extract_code(decoded_code, filepath_temp_code)
    if not isinstance(code_snippet, str):
        code_snippet = ""

    syntax_ok = is_syntax_valid(code_snippet)
    print(f"\n🧪 Syntax Valid: {syntax_ok}")

    # Run unit tests (if provided)
    if test_list and syntax_ok:
        test_result = run_unit_tests(code_snippet, test_list, filepath_test_dir)
        passed = test_result.get("passed", False)
        print(f"\n✅ Unit Tests Passed: {passed}")
    else:
        passed = False
        print(f"\n⚠️ Skipping Unit Tests")

    # Run pylint (assume analyze_code_with_pylint exists)
    pylint_result = analyze_code_with_pylint(code_snippet, filepath_temp_code)
    pylint_score = pylint_result.get("score", 0.0)
    print(f"\n📊 Pylint Score: {pylint_score:.2f}")

    return {
        "generated_code": code_snippet,
        "syntax_valid": syntax_ok,
        "unit_tests_passed": passed,
        "pylint_score": pylint_score
    }


