import re

# def extract_code(text):
#     """
#     Extracts the first Python code block from a string containing markdown-style triple backticks.
#     """
#     if not isinstance(text, str):
#         text = str(text)
    
#     print(text)
#     code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
#     print("_--------------------------------_")
#     print(code_blocks)
#     return code_blocks[0].strip() if code_blocks else None


import re

def extract_code(text):
    """
    Extracts the first Python function code block from the given text.
    The function starts with 'def ' and continues until the first
    unindented line or end of string.

    Returns the function code as a string (including 'def ...').
    """
    if not isinstance(text, str):
        text = str(text)

    # Regex to match a function block:
    # - Start at 'def ' at line start
    # - Capture all indented lines following it (including blank lines)
    pattern = re.compile(
        r"(^def\s.+?:\n"           # function def line (starts with def ...:)
        r"(?:^[ \t]+.*\n)*"        # subsequent indented lines (body)
        r")",
        re.MULTILINE
    )

    matches = pattern.findall(text)
    if matches:
        return matches[0].rstrip()
    else:
        return None
