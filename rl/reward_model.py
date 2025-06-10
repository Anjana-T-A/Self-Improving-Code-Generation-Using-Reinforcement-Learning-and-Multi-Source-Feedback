from utils.sonar_api import analyze_code_with_sonarqube
from utils.rewards import static_analysis_reward
from utils.unit_tests import run_unit_tests

def compute_combined_reward(code, sample, alpha=0.5):
    test_score = run_unit_tests(code, sample)
    sonar_result = analyze_code_with_sonarqube(code)
    sonar_score = static_analysis_reward(sonar_result)

    combined_reward = alpha * test_score + (1 - alpha) * sonar_score
    return combined_reward
