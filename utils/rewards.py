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

def extract_pylint_score(pylint_output):
    """
    Extract the numeric score from the Pylint parseable output.
    Args:
        pylint_output (str): The output from Pylint in parseable format.
    Returns:
        float: The numeric score (0.0 to 10.0) or 0.0 if parsing fails.
    """
    if isinstance(pylint_output, dict):
        return pylint_output.get("score", 0.0)
    
    score_line = next(
        (line for line in pylint_output.splitlines() if "Your code has been rated at" in line),
        None
    )
    score = 0.0  # Default to 0.0 if parsing fails
    if score_line:
        try:
            # Example line: "************* Module generated_code\n...Your code has been rated at 7.50/10"
            raw_score = score_line.split(" ")[6].split("/")[0]
            score = float(raw_score)
        except (IndexError, ValueError):
            print("Failed to parse Pylint score line:", score_line)
    return score