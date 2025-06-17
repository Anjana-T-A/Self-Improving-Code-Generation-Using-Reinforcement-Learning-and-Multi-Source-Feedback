import re

def extract_code(text):
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
    return matches[0].rstrip() if matches else None
