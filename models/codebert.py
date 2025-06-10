import os
from huggingface_hub import InferenceClient


client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

def extract_features(prompt):
    """
    This function sends a prompt to the CodeBERT model (encoder-only),
    and retrieves embeddings (feature extraction) – NOT code generation.
    """
    result = client.feature_extraction(
        prompt,
        model="microsoft/codebert-base",
    )
    print("Feature embeddings shape:", len(result))  # prints length of the embeddings list
    return result
