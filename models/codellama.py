# # # from transformers import AutoTokenizer, AutoModelForCausalLM

# # # model_path = "codellama/CodeLlama-7b-Python-hf"
# # # tokenizer = AutoTokenizer.from_pretrained(model_path)
# # # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# # # def generate_code(prompt):
# # #     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# # #     output = model.generate(**inputs, max_new_tokens=128)
# # #     return tokenizer.decode(output[0], skip_special_tokens=True)

# # from huggingface_hub import InferenceClient

# # # Example: StarCoder

# # # Replace with appropriate repo for each model
# # # MODEL_ID = "codellama/CodeLlama-7b-Python-hf"
# # MODEL_ID ="codeparrot/codeparrot"
# # client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# # def generate_code(prompt):
# #     # Call the text-generation endpoint
# #     response = client.text_generation(
# #         prompt,
# #         max_new_tokens=128,
# #         temperature=0.2,
# #         top_p=0.95
# #     )
# #     return response.generated_text


# from huggingface_hub import InferenceClient
# from data import datasets
# import os
# # Correctly set the model ID and token
# MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-base"

# # Initialize the InferenceClient with the model ID and token
# client = InferenceClient(model=MODEL_ID, token=hf_token)
# # datasets = datasets.load_dataset("mbpp", split="train[:10%]")
# # print(datasets[0])


# client = InferenceClient(
#     provider="sambanova",
#     api_key=hf_token,
# )

# completion = client.chat.completions.create(
#     model="deepseek-ai/DeepSeek-R1-0528",
#     messages=[
#         {
#             "role": "user",
#             "content": "python code to create a list of numbers less than 10. Provide only the code and only once"
#         }
#     ],
# )

# print(completion.choices[0].message)
# # # Generate text
# # response = client.text_generation(
# #     prompt="generate prime numebrs below 10",
# #     max_new_tokens=128,
# #     temperature=0.2,
# #     top_p=0.95
# # )
# # print(response.generated_text)