import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"   # Free via HF Inference API
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"


def call_llm(prompt: str, max_tokens: int = 512) -> str:
    """Call Hugging Face Inference API — free tier."""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "temperature": 0.3},
    }
    response = requests.post(HF_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"].replace(prompt, "").strip()
    return "I'm having trouble connecting right now. Please try again."


def classify_intent(query: str) -> str:
    """Determine which agent should handle this query."""
    prompt = f"""Classify this query into exactly one of: career, project, scheduling, general.
Query: "{query}"
Answer with only the category word:"""
    result = call_llm(prompt, max_tokens=10).lower().strip()
    for category in ["career", "project", "scheduling"]:
        if category in result:
            return category
    return "general"
