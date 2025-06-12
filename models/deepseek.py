# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_path = "deepseek-ai/deepseek-coder-6.7b-base"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# def generate_code(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     output = model.generate(**inputs, max_new_tokens=128)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

from huggingface_hub import InferenceClient
import configparser
# Example: StarCoder
config = configparser.ConfigParser()
config.read('./configs/config.ini')  # Or the path to your config file

# Get the Hugging Face token from the configuration
HF_TOKEN = config['HuggingFace']['token']
# Replace with appropriate repo for each model
MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-base"

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

def generate_code(prompt):
    # Call the text-generation endpoint
    response = client.text_generation(
        prompt,
        max_new_tokens=128,
        temperature=0.2,
        top_p=0.95
    )
    return response.generated_text
