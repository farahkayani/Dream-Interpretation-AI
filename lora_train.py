# -*- coding: utf-8 -*-
"""
LoRA Fine-Tuning on Dream Interpretation Dataset (CPU Safe)
Author: Farah
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ==== Environment ====
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# ==== Config ====
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "lora_corpus.jsonl"

# ==== Load Dataset ====
dataset = load_dataset("json", data_files=DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    inputs = [f"Interpret this dream:\n{i}" for i in example["instruction"]]
    outputs = example["output"]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(outputs, padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize, batched=True)

# ==== Load Model ====
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": "cpu"},
    torch_dtype=torch.float32
)

# ==== Apply LoRA ====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ==== Training Args ====
training_args = TrainingArguments(
    output_dir="./lora_tinyllama_dreams",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch"
)

# ==== Trainer ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"] if "train" in tokenized_dataset else tokenized_dataset
)

trainer.train()

# ==== Save ====
model.save_pretrained("./lora_tinyllama_dreams")
print("✅ LoRA fine-tuning completed and saved.")
