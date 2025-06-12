import os
import ast
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.unit_tests import run_unit_tests # Optional: make sure this is implemented
from typing import List
import random


def compile_check(code: str) -> bool:
    try:
        ast.parse(code)
        compile(code, '<string>', 'exec')
        return True
    except Exception:
        return False


def generate_completions(model, tokenizer, prompt: str, k: int = 10, max_new_tokens: int = 128) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=k,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


def evaluate_model(model_path: str, prompts: List[str], k: int = 10, unit_test_fn=None):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    total_compilable = 0
    total_generated = 0
    total_pass_at_5 = 0
    total_pass_at_10 = 0

    for idx, prompt in enumerate(prompts):
        completions = generate_completions(model, tokenizer, prompt, k=k)

        compilable = [code for code in completions if compile_check(code)]
        pass_tests = [code for code in compilable if unit_test_fn and unit_test_fn(code)] if unit_test_fn else compilable

        if len(pass_tests) >= 5:
            total_pass_at_5 += 1
        if len(pass_tests) >= 10:
            total_pass_at_10 += 1

        total_compilable += len(compilable)
        total_generated += len(completions)

        print(f"Prompt {idx+1}/{len(prompts)}")
        print(f"  - Generated: {len(completions)}")
        print(f"  - Compilable: {len(compilable)}")
        print(f"  - Passed Unit Tests: {len(pass_tests)}")

    compiler_error_rate = 1 - (total_compilable / total_generated)
    pass_at_5 = total_pass_at_5 / len(prompts)
    pass_at_10 = total_pass_at_10 / len(prompts)

    print("\n===== Evaluation Results =====")
    print(f"Compiler Error Rate: {compiler_error_rate:.2%}")
    print(f"Pass@5: {pass_at_5:.2%}")
    print(f"Pass@10: {pass_at_10:.2%}")

