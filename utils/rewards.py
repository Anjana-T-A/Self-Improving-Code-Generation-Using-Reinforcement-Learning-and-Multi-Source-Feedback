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

# def extract_pylint_score(pylint_output):
#     """
#     Extract the numeric score from the Pylint parseable output.
#     Args:
#         pylint_output (str): The output from Pylint in parseable format.
#     Returns:
#         float: The numeric score (0.0 to 10.0) or 0.0 if parsing fails.
#     """
#     if isinstance(pylint_output, dict):
#         return pylint_output.get("score", 0.0)
    
#     score_line = next(
#         (line for line in pylint_output.splitlines() if "Your code has been rated at" in line),
#         None
#     )
#     score = 0.0  # Default to 0.0 if parsing fails
#     if score_line:
#         try:
#             # Example line: "************* Module generated_code\n...Your code has been rated at 7.50/10"
#             raw_score = score_line.split(" ")[6].split("/")[0]
#             score = float(raw_score)
#         except (IndexError, ValueError):
#             print("Failed to parse Pylint score line:", score_line)
#     return score

def extract_pylint_score(pylint_output):
    """
    Enhanced Pylint reward function inspired by RLSQM paper
    Returns reward in [-1, 1] with improved shaping for RL training
    """
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


# def extract_pylint_score(pylint_output):
    """
    Enhanced reward function that considers:
    - Pylint score (primary factor)
    - Presence of errors/fatal messages (heavy penalty)
    - Warning count (moderate penalty)
    - Refactor/convention suggestions (light penalty)
    - Non-linear shaping for better training dynamics
    
    Returns a reward in range [-1, 1] with better granularity
    """
    # Default values if parsing fails
    raw_score = 5.0  # Mid-point as default
    errors = 0
    warnings = 0
    conventions = 0
    refactors = 0
    
    # Parse Pylint output
    if isinstance(pylint_output, dict):
        # Structured format
        raw_score = pylint_output.get("score", 5.0)
        messages = pylint_output.get("messages", [])
        errors = sum(1 for m in messages if m["type"] == "error")
        warnings = sum(1 for m in messages if m["type"] == "warning")
        conventions = sum(1 for m in messages if m["type"] == "convention")
        refactors = sum(1 for m in messages if m["type"] == "refactor")
    else:
        # Text format
        score_line = next(
            (line for line in pylint_output.splitlines() 
             if "Your code has been rated at" in line),
            None
        )
        if score_line:
            try:
                raw_score = float(score_line.split(" ")[6].split("/")[0])
            except (IndexError, ValueError):
                pass
        
        # Count message types
        errors = pylint_output.count("error:")
        warnings = pylint_output.count("warning:")
        conventions = pylint_output.count("convention:")
        refactors = pylint_output.count("refactor:")

    # Clamp score to 0-10 range
    raw_score = max(0.0, min(10.0, raw_score))
    
    # Base reward from score (0-10 → -1 to 1)
    score_reward = (raw_score / 5.0) - 1.0  # Linear mapping
    
    # Penalties for different message types
    error_penalty = -0.5 * min(errors, 4)  # Max -2 for ≥4 errors
    warning_penalty = -0.1 * min(warnings, 10)  # Max -1 for ≥10 warnings
    style_penalty = -0.02 * (conventions + refactors)  # Very light penalty
    
    # Combine components
    combined_reward = (
        score_reward +
        error_penalty +
        warning_penalty +
        style_penalty
    )
    
    # Non-linear shaping
    if combined_reward >= 0:
        # Positive rewards use square root for smoother increase
        shaped_reward = combined_reward ** 0.5
    else:
        # Negative rewards use cubic for sharper penalty
        shaped_reward = -((-combined_reward) ** 0.33)
    
    # Final clamping to [-1, 1] range
    final_reward = max(-1.0, min(1.0, shaped_reward))
    
    # Debug information (optional)
    debug_info = {
        "raw_score": raw_score,
        "errors": errors,
        "warnings": warnings,
        "conventions": conventions,
        "refactors": refactors,
        "score_reward": score_reward,
        "penalties": {
            "errors": error_penalty,
            "warnings": warning_penalty,
            "style": style_penalty
        },
        "shaped_reward": shaped_reward,
        "final_reward": final_reward
    }
    print(f"Reward breakdown: {debug_info}")  # Can be logged instead
    
    return final_reward