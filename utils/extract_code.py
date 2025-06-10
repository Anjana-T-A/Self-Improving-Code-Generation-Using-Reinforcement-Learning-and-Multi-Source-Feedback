import re

def extract_code(text):
    """
    Extracts the first Python code block from a string containing markdown-style triple backticks.
    """
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    print("_--------------------------------_")
    print(code_blocks)
    return code_blocks[0].strip() if code_blocks else None