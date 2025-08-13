# -*- coding: utf-8 -*-
"""
Build FAISS index from data/corpus.jsonl using sentence-transformers.
Author: Farah
"""

import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "meta.pkl"

EMBED_MODEL = "all-MiniLM-L6-v2"

def load_corpus(path):
    docs = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return docs

def build_index(docs):
    embedder = SentenceTransformer(EMBED_MODEL)
    texts = [d["text"] for d in docs if d.get("text")]
    emb = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(INDEX_PATH))

    meta = [{"id": d.get("id", i), "text": d["text"]} for i, d in enumerate(docs) if d.get("text")]
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    if not CORPUS_PATH.exists():
        raise SystemExit(f"Corpus not found at {CORPUS_PATH}.")
    docs = load_corpus(CORPUS_PATH)
    build_index(docs)
    print("✅ FAISS index and metadata built successfully.")
