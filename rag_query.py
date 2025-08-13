# -*- coding: utf-8 -*-
"""
Professional Dream Interpretation AI using RAG + TinyLlama.
Author: Farah
"""

from pathlib import Path
import pickle
import faiss
import torch
import warnings
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === Suppress Logs ===
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# === Paths & Models ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "meta.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# === Helper Functions ===
def load_index():
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def retrieve(query, embedder, index, meta, top_k=1):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    scores, ids = index.search(query_emb, top_k)
    return [meta[i]["text"] for i in ids[0]]

def generate_response(llm, tokenizer, context, query):
    context_str = "\n".join(f"- {ctx}" for ctx in context)
    prompt = (
        f"Dream description:\n{query}\n\n"
        f"Related dreams:\n{context_str}\n\n"
        f"Interpretation:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    output = llm.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# === Load Models ===
base_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
llm = PeftModel.from_pretrained(base_model, "./lora_tinyllama_dreams")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

# === Main Program ===
def main():
    index, meta = load_index()
    embedder = SentenceTransformer(EMBED_MODEL)

    print("\n🌙 Dream Interpretation AI")
    print("Hi! I am a Dream Interpretation AI. \nType your dream description and receive an insightful interpretation of your dream.")
    print("Type 'exit' or 'quit' to close.\n")

    while True:
        query = input("📝 Your dream: ")
        if query.lower() in ["exit", "quit"]:
            print("\n✨ Session ended. Sweet dreams!")
            break

        context = retrieve(query, embedder, index, meta)
        answer = generate_response(llm, tokenizer, context, query)

        print("\n📖 Interpretation:")
        print(answer.strip())
        print("-" * 60)

if __name__ == "__main__":
    main()
