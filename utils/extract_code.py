import re

def extract_code(text, file_name):
    """
    Extracts the first Python function definition (with its body) from text.
    Writes the function (and any imports, moved outside) to file_name, with normalized indentation.
    Returns the function code as a string.
    """
    if not isinstance(text, str):
        text = str(text)

    # Extract from markdown code block if present
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    code_text = code_blocks[0] if code_blocks else text

    lines = code_text.splitlines()
    func_code_lines = []
    in_func = False
    func_indent = None
    function_started = False
    local_imports = []

    for idx, line in enumerate(lines):
        # Detect function start
        match = re.match(r'^([ \t]*)def\s+\w+\s*\(.*?\):', line)
        if match and not in_func:
            in_func = True
            function_started = True
            func_indent = len(match.group(1).replace('\t', '    '))
            func_code_lines.append(line.lstrip())
            continue
        if in_func:
            # Function body lines
            if line.strip() == "":
                func_code_lines.append("")
            else:
                leading_spaces = len(line) - len(line.lstrip(' '))
                if leading_spaces > func_indent:
                    body_line = line[func_indent:] if len(line) > func_indent else line.lstrip()
                    # Extract local imports to promote them
                    if re.match(r"^\s*import\s.+", body_line) or re.match(r"^\s*from\s.+\simport\s.+", body_line):
                        local_imports.append(body_line.strip())
                    else:
                        func_code_lines.append(body_line)
                else:
                    break  # End of function body

    # Collect top-level imports above the function
    top_level_imports = []
    for line in lines:
        if line.strip().startswith("def "):
            break
        if re.match(r"^(import .+|from .+ import .+)$", line.strip()):
            top_level_imports.append(line.strip())

    # Deduplicate imports
    all_imports = list(dict.fromkeys(top_level_imports + local_imports))

    func_code = "\n".join(func_code_lines).rstrip()

    # Write to file
    try:
        with open(file_name, 'w') as f:
            if all_imports:
                f.write("\n".join(all_imports) + "\n\n")
            f.write(func_code + "\n")
    except IOError as e:
        print(f"Error writing to file {file_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return func_code