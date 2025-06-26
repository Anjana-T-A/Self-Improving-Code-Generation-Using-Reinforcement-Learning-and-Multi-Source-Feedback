# import re

# def extract_code(text,file_name):
#     """
#     Extracts Python code from either:
#     1. Markdown-style triple-backtick code blocks (```python ... ```)
#     2. Or standalone Python function definitions in raw text (starting with 'def ')

#     Returns the first detected function definition as a string.
#     """
#     if not isinstance(text, str):
#         text = str(text)

#     # First, try to extract from a markdown code block
#     code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
#     try:
#         if code_blocks:
#             code_text = code_blocks[0]
#         else:
#             code_text = text  # Fall back to original text if no code block found
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

#     # Extract the first function definition from the code text
#     pattern = re.compile(
#         r"(^def\s.+?:\n"        # def line
#         r"(?:^[ \t]+.*\n?)*)"   # indented block
#         , re.MULTILINE
#     )

#     matches = pattern.findall(code_text)
#     if matches:
#         try:
#             with open(file_name, 'w') as f:
#                 f.write(matches[0].rstrip())
#         except IOError as e:
#             print(f"Error writing to file {file_name}: {e}")
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")
#         return matches[0].rstrip()
#     else:
#         return ""


import re

# def extract_code(text, file_name):
#     """
#     Extracts Python code from either:
#     1. Markdown-style triple-backtick code blocks (```python ... ```)
#     2. Or standalone Python function definitions in raw text (starting with 'def ')

#     Returns the first detected function definition as a string,
#     and writes imports + function code to the given file.
#     """
#     if not isinstance(text, str):
#         text = str(text)

#     # Extract markdown code blocks or fallback to full text
#     code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
#     code_text = code_blocks[0] if code_blocks else text

#     # Extract imports before first function
#     func_pattern = re.compile(
#         r"(^def\s.+?:\n(?:^[ \t]+.*\n?)*)", re.MULTILINE
#     )
#     func_match = func_pattern.search(code_text)

#     imports = []
#     if func_match:
#         func_start = func_match.start()
#         before_func = code_text[:func_start]
#         # Find import statements before the function
#         imports = re.findall(r"^(import .+|from .+ import .+)$", before_func, re.MULTILINE)
#         func_code = func_match.group().rstrip()
#     else:
#         # No function found, fallback to empty
#         func_code = ""

#     # Write imports and function code to file
#     try:
#         with open(file_name, 'w') as f:
#             if imports:
#                 f.write("\n".join(imports) + "\n\n")
#             f.write(func_code)
#     except IOError as e:
#         print(f"Error writing to file {file_name}: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

#     return func_code



# def extract_code(text, file_name):
#     """
#     Extracts the first Python function definition (with its body) from text.
#     Writes the function (and any imports above it) to file_name.
#     Returns the function code as a string.
#     """
#     if not isinstance(text, str):
#         text = str(text)

#     # Try to extract from markdown code block first
#     code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
#     code_text = code_blocks[0] if code_blocks else text

#     lines = code_text.splitlines()
#     func_code_lines = []
#     in_func = False
#     func_indent = None

#     for idx, line in enumerate(lines):
#         # Detect function start
#         match = re.match(r'^([ \t]*)def\s+\w+\s*\(.*?\):', line)
#         if match and not in_func:
#             in_func = True
#             func_indent = len(match.group(1).replace('\t', '    '))
#             func_code_lines.append(line)
#             continue
#         if in_func:
#             # Check if line is indented more than function or is blank
#             if line.strip() == "":
#                 func_code_lines.append(line)
#             else:
#                 leading_spaces = len(line) - len(line.lstrip(' '))
#                 if leading_spaces > func_indent:
#                     func_code_lines.append(line)
#                 else:
#                     break  # End of function body

#     func_code = "\n".join(func_code_lines).rstrip()

#     # Collect imports before the function
#     imports = []
#     for line in lines:
#         if line.strip().startswith("def "):
#             break
#         if re.match(r"^(import .+|from .+ import .+)$", line.strip()):
#             imports.append(line.strip())

#     # Write to file
#     try:
#         with open(file_name, 'w') as f:
#             if imports:
#                 f.write("\n".join(imports) + "\n\n")
#             f.write(func_code)
#     except IOError as e:
#         print(f"Error writing to file {file_name}: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

#     return func_code

import re

def extract_code(text, file_name):
    """
    Extracts the first Python function definition (with its body) from text.
    Writes the function (and any imports above it) to file_name, with normalized indentation.
    Returns the function code as a string.
    """
    if not isinstance(text, str):
        text = str(text)

    # Try to extract from markdown code block first
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    code_text = code_blocks[0] if code_blocks else text

    lines = code_text.splitlines()
    func_code_lines = []
    in_func = False
    func_indent = None

    for idx, line in enumerate(lines):
        # Detect function start
        match = re.match(r'^([ \t]*)def\s+\w+\s*\(.*?\):', line)
        if match and not in_func:
            in_func = True
            func_indent = len(match.group(1).replace('\t', '    '))
            func_code_lines.append(line.lstrip())
            continue
        if in_func:
            # Check if line is indented more than function or is blank
            if line.strip() == "":
                func_code_lines.append("")
            else:
                leading_spaces = len(line) - len(line.lstrip(' '))
                if leading_spaces > func_indent:
                    # Dedent relative to function indentation
                    func_code_lines.append(line[func_indent:] if len(line) > func_indent else line.lstrip())
                else:
                    break  # End of function body

    func_code = "\n".join(func_code_lines).rstrip()

    # Collect imports before the function
    imports = []
    for line in lines:
        if line.strip().startswith("def "):
            break
        if re.match(r"^(import .+|from .+ import .+)$", line.strip()):
            imports.append(line.strip())

    # Write to file
    try:
        with open(file_name, 'w') as f:
            if imports:
                f.write("\n".join(imports) + "\n\n")
            f.write(func_code)
    except IOError as e:
        print(f"Error writing to file {file_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return func_code