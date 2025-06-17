from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "Qwen/Qwen2.5-Coder-1.5B"  # Change for each model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)
