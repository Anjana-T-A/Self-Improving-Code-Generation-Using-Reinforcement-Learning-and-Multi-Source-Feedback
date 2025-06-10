import os
import subprocess
import json

def static_analysis_reward_sonar(sonar_result):
    print(sonar_result)
    metrics = sonar_result.get('component', {}).get('measures', [])
    if not metrics:
        return 0.0

    reward_sum = 0
    count = 0
    for m in metrics:
        try:
            rating = float(m['value'])
            reward_sum += (5 - rating) / 4
            count += 1
        except (KeyError, ValueError, TypeError):
            continue

    return reward_sum / count if count else 0.0


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
    with open(file_name, 'w') as f:
        f.write(code_snippet)

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

    # Debug output
    print(f"Pylint score: {score:.2f} / 10.0")
    print(f"Normalized reward: {normalized_reward:.2f}")

    return {
        "pylint_messages": lint_output,
        "score": score,
        "normalized_reward": normalized_reward
    }
