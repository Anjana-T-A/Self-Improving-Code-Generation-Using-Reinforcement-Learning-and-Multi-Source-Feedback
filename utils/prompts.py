import re

def create_prompt(problem, test_cases):
    func_name = extract_function_name(test_cases)
    prompt = f"""### Text:
{problem}

### Your task:
Write a Python function named `{func_name}` that solves the problem described above. 
Do not include any test cases or example usage. Only return the complete function definition.
"""
    return prompt


def extract_function_name(test_cases):
    for case in test_cases:
        match = re.search(r'assert\s+([a-zA-Z_]\w*)\s*\(', case)
        if match:
            return match.group(1)
    return None
