import configparser
from huggingface_hub import InferenceClient

# Load the configuration from the config.ini file
config = configparser.ConfigParser()
config.read('./configs/config.ini')  # Or the path to your config file

# Get the Hugging Face token from the configuration
HF_TOKEN = config['HuggingFace']['token']
client = InferenceClient(
    provider="nscale",
    api_key=HF_TOKEN,
)

def generate_code(prompt):
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=512,
        top_p=0.95
    )

    return completion.choices[0].message.content