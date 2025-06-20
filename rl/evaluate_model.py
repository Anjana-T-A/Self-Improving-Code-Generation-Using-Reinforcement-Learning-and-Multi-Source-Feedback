import ast
import torch
from data.datasets import get_dataloader
from rl.ppo_trainer_batch import collate_batch
from utils.extract_code import extract_code
from utils.unit_tests import run_unit_tests
from torch.utils.data import DataLoader

def is_syntax_valid(code: str) -> bool:
    """Check if the code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def evaluate_pass_at_k(model, tokenizer, test_dataset, device, k_values=[1,5,10]):
    total_samples = 0
    syntax_valid_counts = 0
    pass_at_k_counts = {k: 0 for k in k_values}
    dataloader = get_dataloader(test_dataset, tokenizer, device)
    print("\n===== Evaluating Pass@k and Compiler Error Rate =====")
 
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, prompts, raw_batch) in enumerate(dataloader):
            batch_size = len(prompts)
            total_samples += batch_size

            for i in range(batch_size):
                prompt_input_ids = input_ids[i].unsqueeze(0).to(device)
                prompt_attention_mask = attention_mask[i].unsqueeze(0).to(device)

                # Generate k samples per prompt
                all_pass = []
                all_syntax = []

                for _ in range(max(k_values)):
                    outputs = model.generate(
                        input_ids=prompt_input_ids,
                        attention_mask=prompt_attention_mask,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    prompt_len = prompt_input_ids.shape[-1]
                    gen_tokens = outputs[0][prompt_len:]
                    decoded_code = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    print(f"\n[🔹 Prompt #{total_samples}-{i}]")
                    print(f"🟡 Prompt:\n{prompts[i]}")
                    print(f"🟢 Generated:\n{decoded_code}")
                    code_snippet = extract_code(decoded_code,"test_generate.py")
                    if not isinstance(code_snippet, str):
                        # print+(f"⚠️ [Invalid code snippet] Got {type(code_snippet)}: {repr(code_snippet)}")
                        code_snippet = "" 
                    # print(f"🔵 Extracted Code:\n{code_snippet}")
                    syntax_ok = is_syntax_valid(code_snippet)
                    all_syntax.append(syntax_ok)

                    if syntax_ok:
                        passed = run_unit_tests(code_snippet,raw_batch[i]["test_list"],"test_code.py")
                    else:
                        passed = False
                    all_pass.append(passed)

                syntax_valid_counts += sum(all_syntax)

                # Calculate pass@k for each k
                for k in k_values:
                    # pass@k means at least one out of k samples passes unit tests
                    if any(all_pass[:k]):
                        pass_at_k_counts[k] += 1

    num_prompts = total_samples
    compiler_error_rate = 1 - (syntax_valid_counts / (num_prompts * max(k_values)))

    print(f"Total prompts evaluated: {num_prompts}")
    print(f"Compiler Error Rate: {compiler_error_rate:.3f}")

    for k in k_values:
        pass_at_k = pass_at_k_counts[k] / num_prompts
        print(f"Pass@{k}: {pass_at_k:.3f}")

    return {
        "compiler_error_rate": compiler_error_rate,
        **{f"pass@{k}": pass_at_k_counts[k] / num_prompts for k in k_values}
    }
