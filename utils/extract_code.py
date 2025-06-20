import re

def extract_code(text,file_name):
    """
    Extracts Python code from either:
    1. Markdown-style triple-backtick code blocks (```python ... ```)
    2. Or standalone Python function definitions in raw text (starting with 'def ')

    Returns the first detected function definition as a string.
    """
    if not isinstance(text, str):
        text = str(text)

    # First, try to extract from a markdown code block
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        code_text = code_blocks[0]
    else:
        code_text = text  # Fall back to original text if no code block found

    # Extract the first function definition from the code text
    pattern = re.compile(
        r"(^def\s.+?:\n"        # def line
        r"(?:^[ \t]+.*\n?)*)"   # indented block
        , re.MULTILINE
    )

    matches = pattern.findall(code_text)
    try:
        with open(file_name, 'w') as f:
            f.write(matches[0].rstrip())
    except IOError as e:
        print(f"Error writing to file {file_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return matches[0].rstrip() if matches else ""



"""
import re

def extract_code(text):
    
    if not isinstance(text, str):
        text = str(text)

    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if not code_blocks:
        code_blocks = [text]  # no blocks, treat full text as one block

    def block_has_function_with_return_at_end(block):
        # find first function
        m = re.search(r"(^def\s.+?:\n(?:^[ \t]+.*\n?)*)", block, re.MULTILINE)
        if not m:
            return False
        func_code = m.group()
        # Check if last non-empty line inside function starts with 'return'
        func_lines = func_code.strip().splitlines()
        # Skip first line (def ...) and find last meaningful line
        body_lines = func_lines[1:]
        # Reverse iterate to find last non-empty line
        for line in reversed(body_lines):
            if line.strip():
                return line.strip().startswith("return")
        return False

    # Select block with function ending with return if exists
    selected_block = None
    for block in code_blocks:
        if block_has_function_with_return_at_end(block):
            selected_block = block
            break
    if selected_block is None:
        selected_block = code_blocks[0]

    # Extract first function from selected block
    func_match = re.search(
        r"(^def\s.+?:\n(?:^[ \t]+.*\n?)*)", selected_block, re.MULTILINE
    )
    if not func_match:
        return None

    func_start = func_match.start()
    func_code = func_match.group()

    before_func = selected_block[:func_start]
    imports = re.findall(r"^(import .+|from .+ import .+)$", before_func, re.MULTILINE)

    parts = []
    if imports:
        parts.append("\n".join(imports))
    parts.append(func_code.rstrip())

    return "\n\n".join(parts)
"""