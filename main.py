import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from data.datasets import load_datasets
from rl.ppo_trainer import run_ppo_training
from rl.ppo_trainer_batch import run_ppo_epoch_training
datasets = load_datasets()
model_checkpoint = "ppo_checkpoints"

# 
# run_ppo_training("codellama/CodeLlama-7b-Python-hf", datasets["train"])
# run_ppo_training("Salesforce/codegen-2B-multi", datasets["train"],100)
# run_ppo_training("bigcode/starcoder2-3b", datasets["train"],100)
# run_ppo_batch_training("bigcode/starcoder2-3b", datasets["train"],100)
run_ppo_epoch_training("deepseek-ai/deepseek-coder-1.3b-instruct",output_dir="./ppo_checkpoints", 
    batch_size=2, 
    num_epochs=3,  # Added epoch control
    eval_every=1  )




# # evaluate_model(model_checkpoint, datasets["test"], k=10, unit_test_fn=None)




# def evaluate_dataset_static_analysis(dataset_split, model_module):
#     rewards = []
#     for item in dataset_split:
#         prompt = create_prompt(item['text'], item['code'])
#         response = model_module.generate_code(prompt)
#         code = extract_code(response)
#         # test = analyze_code_with_pylint(code)
#         # sonar_result = analyze_code_with_sonarqube(code)
#         # reward = static_analysis_reward(sonar_result)
#         # rewards.append({
#         #     "prompt": prompt,
#         #     "code": code,
#         #     "reward": reward
#         # })
#         print( f"====Prompt=====\n {prompt}\n =====response======: {response}")

# evaluate_dataset_static_analysis(datasets['train'], starcoder)