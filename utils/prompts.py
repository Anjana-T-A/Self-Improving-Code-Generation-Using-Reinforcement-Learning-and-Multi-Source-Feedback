import re

def create_prompt(problem, test_cases):
    func_name = extract_function_name(test_cases)
    prompt = f""" Problem:
{problem} \n
Generate a python code to solve not in any other languages.
Give a python solution for above problem with given function name  "{func_name}"
Do not provide function calls or examples.
"""

    return prompt
# def create_prompt(problem, test_cases):
#     func_name = extract_function_name(test_cases)
#     prompt = f"""### Text:
# {problem}

# Write only the Python function named `{func_name}`. Do NOT include any explanations, comments, examples, or docstrings. Just the function definition and its code.
# DO NOT Provide function calls.
# """
#     return prompt

def extract_function_name(test_cases):
    for case in test_cases:
        match = re.search(r'assert\s+([a-zA-Z_]\w*)\s*\(', case)
        if match:
            return match.group(1)
    return None
