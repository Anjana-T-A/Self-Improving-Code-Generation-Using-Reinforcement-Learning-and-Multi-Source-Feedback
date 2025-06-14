import re

def create_prompt(problem, test_cases):
    func_name = extract_function_name(test_cases)
    prompt = f"""### Text:
{problem}

Write only the Python function named m`{func_name}`n_length that finds the longest chain from a given list of pairs. Do NOT include any explanations, comments, examples, or docstrings. Just the function definition and its code.

"""
    return prompt


def extract_function_name(test_cases):
    for case in test_cases:
        match = re.search(r'assert\s+([a-zA-Z_]\w*)\s*\(', case)
        if match:
            return match.group(1)
    return None
