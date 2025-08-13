# -*- coding: utf-8 -*-
"""
prepare_lora_dataset.py
Convert dream corpus into LoRA fine-tuning dataset format.
Author: Farah
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CORPUS_FILE = DATA_DIR / "corpus.jsonl"
LORA_FILE = BASE_DIR / "lora_corpus.jsonl"

if not CORPUS_FILE.exists():
    raise FileNotFoundError(f"Corpus file not found at {CORPUS_FILE}")

lora_data = []

# Load corpus
with open(CORPUS_FILE, "r", encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        dream_text = entry.get("text", "").strip()
        if not dream_text:
            continue

        # Convert to LoRA fine-tuning format
        lora_data.append({
            "instruction": dream_text,
            "input": "",
            "output": "Provide interpretation or summary here"
        })

# Save LoRA dataset
with open(LORA_FILE, "w", encoding="utf-8") as f:
    for item in lora_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(lora_data)} entries to {LORA_FILE}")
