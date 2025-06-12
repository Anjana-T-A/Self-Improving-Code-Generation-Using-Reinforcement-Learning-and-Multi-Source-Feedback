from transformers import AutoTokenizer, AutoModelForCausalLM
import configparser

config = configparser.ConfigParser()
config.read('./configs/config.ini')  # Or the path to your config file

# Get the Hugging Face token from the configuration
HF_TOKEN = config['HuggingFace']['token']

model_path = "bigcode/starcoderbase"  # Change for each model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)
