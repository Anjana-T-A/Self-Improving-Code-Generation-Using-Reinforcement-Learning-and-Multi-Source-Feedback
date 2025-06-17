import os
import subprocess
import json

# def analyze_code_with_pylint(code_snippet, file_name="generated_code.py"):
#     # Write code to file
#     with open(file_name, 'w') as f:
#         f.write(code_snippet)

#     # Run pylint and get JSON output
#     result = subprocess.run(
#         ["pylint", "--output-format=json", file_name],
#         capture_output=True,
#         text=True
#     )

#     # Parse JSON output (even if linting returned non-zero exit code)
#     try:
#         lint_output = json.loads(result.stdout)
#     except json.JSONDecodeError:
#         print("Failed to parse pylint output:")
#         print(result.stdout)
#         raise

#     # Get numeric score
#     score_result = subprocess.run(
#         ["pylint", file_name, "-f", "parseable"],
#         capture_output=True,
#         text=True
#     )
#     score_line = next((line for line in score_result.stdout.splitlines() if "Your code has been rated at" in line), None)
#     score = None
#     if score_line:
#         score = score_line.split(" ")[6].split("/")[0]

#     print ("score", score)
#     return {
#         "pylint_messages": lint_output,
#         "score": score
#     }


import os
import subprocess
import json

def analyze_code_with_pylint(code_snippet, file_name="generated_code.py"):
    """
    Analyze the generated Python code using Pylint and compute a normalized reward.
    
    Args:
        code_snippet (str): The code to analyze.
        file_name (str): Filename to write the code to (default: 'generated_code.py').

    Returns:
        dict: {
            'pylint_messages': list of Pylint messages (JSON),
            'score': numeric score (0.0 to 10.0),
            'normalized_reward': float (0.0 to 1.0)
        }
    """
    # Write the generated code to a file
    try:
        with open(file_name, 'w') as f:
            f.write(code_snippet)
    except IOError as e:
        print(f"Error writing to file {file_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Run Pylint with JSON output to capture detailed lint messages
    result = subprocess.run(
        ["pylint", "--output-format=json", file_name],
        capture_output=True,
        text=True
    )

    try:
        lint_output = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Failed to parse Pylint output as JSON:")
        print(result.stdout)
        lint_output = []

    # Run Pylint again to extract the numeric score from parseable output
    score_result = subprocess.run(
        ["pylint", file_name, "-f", "parseable"],
        capture_output=True,
        text=True
    )
    score_line = next(
        (line for line in score_result.stdout.splitlines() if "Your code has been rated at" in line),
        None
    )

    score = 0.0  # Default to 0.0 if parsing fails
    if score_line:
        # Example line: "************* Module generated_code\n...Your code has been rated at 7.50/10"
        try:
            raw_score = score_line.split(" ")[6].split("/")[0]
            score = float(raw_score)
        except (IndexError, ValueError):
            print("Failed to parse Pylint score line:", score_line)

    # Normalize score to [0.0, 1.0]
    normalized_reward = max(min(score / 10.0, 1.0), 0.0)

    return {
        "messages": lint_output,
        "score": score,
        "normalized_reward": normalized_reward
    }
