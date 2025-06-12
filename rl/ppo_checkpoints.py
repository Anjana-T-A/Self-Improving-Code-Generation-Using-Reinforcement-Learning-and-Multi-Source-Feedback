from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./ppo_checkpoints")
tokenizer = AutoTokenizer.from_pretrained("./ppo_checkpoints")
