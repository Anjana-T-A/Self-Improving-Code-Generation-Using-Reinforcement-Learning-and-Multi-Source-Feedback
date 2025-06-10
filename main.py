from data.datasets import load_datasets
from utils.prompts import create_prompt
from utils.sonar_api import analyze_code_with_sonarqube
from utils.rewards import static_analysis_reward_sonar
from utils.extract_code import extract_code
import models.starcoder as starcoder
import models.deepseek as deepseek
import models.qwen as qwen
from utils.pylint import analyze_code_with_pylint

datasets = load_datasets()

def evaluate_dataset_static_analysis(dataset_split, model_module):
    rewards = []
    for item in dataset_split:
        prompt = create_prompt(item['text'])
        print(prompt)
        response = model_module.generate_code(prompt)
        code = extract_code(response)
        test = analyze_code_with_pylint(code)
        # sonar_result = analyze_code_with_sonarqube(code)
        # reward = static_analysis_reward(sonar_result)
        # rewards.append({
        #     "prompt": prompt,
        #     "code": code,
        #     "reward": reward
        # })
    return rewards


results = evaluate_dataset_static_analysis(datasets['train'], qwen)

print(results[:2])


# from data.datasets import load_datasets
# from utils.prompts import create_prompt
# from utils.sonar_api import analyze_code_with_sonarqube
# from utils.rewards import static_analysis_reward
# from utils.unit_tests import run_unit_tests
# import models.starcoder as starcoder
# import models.deepseek as deepseek
# import models.codellama as codellama

# datasets = load_datasets()

# def evaluate_dataset_with_multisource_reward(dataset_split, model_module):
#     rewards = []
#     for item in dataset_split:
#         prompt = create_prompt(item['text'])
#         code = model_module.generate_code(prompt)
        
#         # 1️⃣ Static analysis reward
#         sonar_result = analyze_code_with_sonarqube(code)
#         reward_static = static_analysis_reward(sonar_result)
        
#         # 2️⃣ Unit test reward (if test cases exist in dataset)
#         test_cases = item.get('test_list', '') or item.get('test', '')
#         unit_test_result = run_unit_tests(code, test_cases)
#         reward_unit_test = unit_test_result['pass_rate']
        
#         # 3️⃣ Combine rewards (simple average, or tune α)
#         alpha = 0.5  # weight for unit tests
#         reward = alpha * reward_unit_test + (1 - alpha) * reward_static

#         rewards.append({
#             "prompt": prompt,
#             "code": code,
#             "reward_unit_test": reward_unit_test,
#             "reward_static": reward_static,
#             "reward_combined": reward
#         })
#     return rewards

# # Example: Evaluate MBPP train split with StarCoder
# results = evaluate_dataset_with_multisource_reward(datasets['train'], starcoder)

# # Debug: Print first 2 results
# print(results[:2])



# def evaluate_dataset_unit_test_only(dataset_split, model_module):
#     results = []
#     for item in dataset_split:
#         prompt = create_prompt(item['text'])
#         code = model_module.generate_code(prompt)
        
#         # Extract unit tests
#         test_cases = item.get('test_list', '') or item.get('test', '')
#         unit_test_result = run_unit_tests(code, test_cases)
#         reward = unit_test_result['pass_rate']
        
#         results.append({
#             "prompt": prompt,
#             "code": code,
#             "unit_test_result": unit_test_result,
#             "reward": reward
#         })
#     return results

# # Example: Evaluate MBPP train split with StarCoder
# results = evaluate_dataset_unit_test_only(datasets['train'], starcoder)

# # Debug: Print first 2 results
# print(results[:2])

# from data.datasets import load_datasets
# from utils.prompts import create_prompt
# from utils.sonar_api import analyze_code_with_sonarqube
# from utils.rewards import static_analysis_reward
# from utils.unit_tests import run_unit_tests
# from utils.extract_code import extract_code

# # Dynamically imported model module will provide `generate_code(prompt)`
# def evaluate_model(dataset_split, model_module, mode="multi"):
#     results = []
#     for item in dataset_split:
#         prompt = create_prompt(item['text'])
#         response = model_module.generate_code(prompt)
#         code = extract_code(response)

#         test_cases = item.get('test_list', '') or item.get('test', '')

#         reward_unit_test, reward_static, reward_combined = 0.0, 0.0, 0.0

#         if mode == "unit":
#             test_result = run_unit_tests(code, test_cases)
#             reward_unit_test = test_result['pass_rate']
#             reward = reward_unit_test

#         elif mode == "static":
#             sonar_result = analyze_code_with_sonarqube(code)
#             reward_static = static_analysis_reward(sonar_result)
#             reward = reward_static

#         elif mode == "multi":
#             sonar_result = analyze_code_with_sonarqube(code)
#             reward_static = static_analysis_reward(sonar_result)

#             test_result = run_unit_tests(code, test_cases)
#             reward_unit_test = test_result['pass_rate']

#             alpha = 0.5
#             reward = alpha * reward_unit_test + (1 - alpha) * reward_static

#         else:
#             raise ValueError("Unsupported mode: choose from ['unit', 'static', 'multi']")

#         results.append({
#             "prompt": prompt,
#             "code": code,
#             "reward_unit_test": reward_unit_test,
#             "reward_static": reward_static,
#             "reward_combined": reward
#         })

#     return results
