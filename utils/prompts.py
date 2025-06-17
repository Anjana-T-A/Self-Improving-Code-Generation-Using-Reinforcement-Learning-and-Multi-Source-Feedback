import re

def create_prompt(problem, test_cases):
    func_name = extract_function_name(test_cases)
    prompt = f""" 
Problem: {problem} \n

Give the python code for above problem with function name  "{func_name}"
Like complete the code below:
def {func_name}

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
