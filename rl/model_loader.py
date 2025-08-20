from transformers import AutoModelForCausalLM, AutoTokenizer

def load_checkpoint(checkpoint_path, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, local_files_only=True)
    model.to(device)
   
    return model, tokenizer
