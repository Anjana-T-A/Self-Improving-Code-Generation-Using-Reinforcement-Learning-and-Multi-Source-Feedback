def extract_pylint_score(pylint_output):
    """
    Extracts and shapes a reward based on Pylint output.

    Args:
        pylint_output: The output from Pylint (dict or string).

    Returns:
        A reward value in the range [-1, 1].
    """
    if pylint_output is None:
        print("Error: Pylint output is None!")
        return -1.0  # Or some other appropriate penalty

    # ... rest of your code ...
    # 1. Syntax Check (Critical Failure)
    if "syntax-error" in str(pylint_output).lower():
        return -1.0  # Immediate strong penalty for syntax errors

    # 2. Extract Raw Score
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
            if score_line:  # Check if score_line is not empty
                raw_score = float(score_line.split("/")[0].split()[-1])
        except (ValueError, IndexError):
            print("Warning: Could not parse Pylint score.")  # Log the error
            return -0.5  # Moderate penalty if parsing fails

    # 3. Quality-Based Shaping (Paper-Inspired)
    # Map to [-1, 1] with non-linear transformation
    if raw_score >= 7.0:  # High quality
        reward = min(1.0, (raw_score - 7.0) / 3.0)  # [7,10] -> [0,1]
    elif raw_score >= 5.0:  # Acceptable
        reward = (raw_score - 5.0) / 2.0 - 0.5  # [5,7) -> [-0.5,0.5]
    else:  # Low quality
        reward = -0.5 + (raw_score / 5.0)/2  # [0,5) -> [-0.5,0)

    # 4. Penalty Amplification for Critical Issues
    if "fatal" in str(pylint_output).lower():
        reward = max(-1.0, reward - 0.3)

    return reward

def extract_unit_test_score(test_result_output):
    """
    Compute a fully normalized reward score in [-1, 1] from unit test results.
    """
    # Initialize rewards
    reward_coarse = 0.0
    reward_fine = 0.0
    reward_adaptive = 0.0

    # Count passed/failed tests
    passed = test_result_output.get("passed", 0)
    failed = test_result_output.get("failed", 0)
    total = passed + failed

    # ---- Coarse-Grained Feedback ----
    if "SyntaxError" in test_result_output:
        reward_coarse = -1.0
    elif failed > 0:
        reward_coarse = -0.3
    elif passed > 0 and failed == 0:
        reward_coarse = 1.0

    # ---- Fine-Grained Feedback ----
    error_weights = {
        "IndexError": -0.2,
        "TypeError": -0.3,
        "ValueError": -0.2,
        "NameError": -0.5,
        "KeyError": -0.2,
        "ZeroDivisionError": -0.4,
        "ImportError": -0.3,
        "EOFError": -0.2,
        "TimeoutError": -0.4,
        "IndentationError": 0.0,   # Ignored
        "Triple-quoted": 0.0       # Ignored
    }

    fine_penalties = [
        weight for err, weight in error_weights.items()
        if err in test_result_output
    ]
    reward_fine = sum(fine_penalties)
    reward_fine = max(-1.0, reward_fine)  # Clip lower bound

    # ---- Adaptive Feedback ----
    if total > 0:
        reward_adaptive = -0.3 + 1.3 * (passed / total)
    else:
        reward_adaptive = -0.3  # No tests run

    # ---- Weighted Combination ----
    weight_coarse = 1.0
    weight_fine = 0.5
    weight_adaptive = 2.0
    total_weight = weight_coarse + weight_fine + weight_adaptive

    combined_reward = (
        weight_coarse * reward_coarse +
        weight_fine * reward_fine +
        weight_adaptive * reward_adaptive
    ) / total_weight

    # ---- Full Normalization to [-1, 1] ----
    theoretical_min = (
        weight_coarse * -1.0 + weight_fine * -1.0 + weight_adaptive * -0.3
    ) / total_weight  # = -0.642857...
    
    theoretical_max = (
        weight_coarse * 1.0 + weight_fine * 0.0 + weight_adaptive * 1.0
    ) / total_weight  # = 0.857143...

    # Apply linear normalization
    normalized_reward = 2 * (combined_reward - theoretical_min) / (theoretical_max - theoretical_min) - 1

    return normalized_reward
