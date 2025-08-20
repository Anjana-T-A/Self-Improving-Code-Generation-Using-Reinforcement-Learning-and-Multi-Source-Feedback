import re

def extract_function_signature(code_string):
    """
    Extracts the function heading (def + function name + arguments + colon)
    from a Python code string.

    Args:
        code_string (str): A string containing Python code.

    Returns:
        str: The function heading, or None if no function heading is found.
    """
    pattern = r"def\s+(\w+)\s*\(([^)]*)\)\s*:"
    match = re.search(pattern, code_string)
    if match:
        return match.group(0)  # Return the entire matched function heading
    else:
        return None

def extract_function_name(code):
    pattern = r"def\s+(\w+)\s*\("
    match = re.search(pattern, code)
    if match:
        return match.group(1)  # Return the function name
    else:
        return None
    
def create_prompt(problem, code):
    func= extract_function_signature(code)  # Assuming this function exists and works
    func_name = extract_function_name(code)
    prompt = f"""
            Problem: 

            Write a Python function called `{func_name}` that solves the following problem:

            {problem}
            DO NOT repeat the prompt in response
            The function should have the following signature:

            {func}

            No comments needed in response and only must provide the whole code in response.
            """
    
    return prompt