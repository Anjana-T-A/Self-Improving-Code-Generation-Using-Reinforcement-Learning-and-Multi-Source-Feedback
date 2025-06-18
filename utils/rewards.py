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
    # 1. Syntax check (critical failure)
    if "syntax-error" in str(pylint_output):
        return -1.0  # Immediate strong penalty for syntax errors

    # 2. Extract raw score
    raw_score = 0.0
    if isinstance(pylint_output, dict):
        raw_score = pylint_output.get("score", 0.0)
    else:
        try:
            # More robust parsing of Pylint output
            score_line = next(
                (line for line in pylint_output.splitlines() 
                 if "rated at" in line),
                ""
            )
            raw_score = float(score_line.split("/")[0].split()[-1])
        except (ValueError, IndexError):
            pass  # Default remains 0.0

    # 3. Quality-based shaping (paper-inspired)
    # Map to [-1, 1] with non-linear transformation
    if raw_score >= 7.0:  # High quality
        reward = min(1.0, (raw_score - 7.0) / 3.0)  # [7,10] -> [0,1]
    elif raw_score >= 5.0:  # Acceptable
        reward = (raw_score - 5.0) / 4.0  # [5,7) -> [0,0.5]
    else:  # Low quality
        reward = -1.0 + (raw_score / 5.0)  # [0,5) -> [-1,0)

    # 4. Penalty amplification for critical issues
    if "fatal" in str(pylint_output).lower():
        reward = max(-1.0, reward - 0.3)
        
    return reward

