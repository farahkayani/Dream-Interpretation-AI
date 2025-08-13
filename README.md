---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
---

# Model Card – Dream Interpretation AI (RAG + LoRA)

This model combines Retrieval-Augmented Generation (RAG) with Low-Rank Adaptation (LoRA) fine-tuning to deliver accurate, culturally aware, and context-driven dream interpretations.

---

## Model Details

### Model Description
The Dream Interpretation AI is designed to interpret user-described dreams by leveraging semantically similar past cases retrieved from a FAISS-based vector database and using them as context for a fine-tuned TinyLlama model. LoRA fine-tuning adapts the model’s attention layers for domain-specific reasoning while maintaining computational efficiency.

- **Developed by:** Farah Asaad Kayani
- **Model type:** Retrieval-Augmented Generation + LoRA Fine-Tuned LLM
- **Language(s):** English (can be adapted to other languages)
- **License:** MIT
- **Finetuned from model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0

### Model Sources
- **Repository:** *(link to your GitHub repo)*
- **Demo:** *(link if available)*

---

## Uses

### Direct Use
- Personal dream analysis
- Educational purposes in psychology or cultural studies
- AI-based narrative interpretation

### Downstream Use
- Integration into chatbots or wellness applications
- Research in AI-assisted mental health tools

### Out-of-Scope Use
- Medical or psychiatric diagnosis
- Legal or financial decision-making

---

## Bias, Risks, and Limitations
- Cultural interpretations may not match all users’ backgrounds.
- The model may reflect biases present in the training data.
- Not a substitute for professional mental health advice.

**Recommendations:** Always clarify that outputs are for informational purposes only.

---

## How to Get Started

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_model = "path/to/your/model"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, lora_model)

prompt = "I dreamed I was flying over mountains."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
Training Details
Training Data
Curated dream corpus with culturally diverse examples

Preprocessed with Sentence Transformers for semantic similarity

Training Procedure
RAG pipeline retrieves top-k semantically similar dreams

LoRA fine-tuning applied to model’s attention layers

Mixed-precision training for efficiency

Evaluation
Testing Data
Separate test set of unseen dream narratives

Metrics
Semantic similarity score (Cosine similarity)

Human evaluation for cultural relevance

Environmental Impact
Hardware Type: A100 GPU

Hours used: ~6 hours

Cloud Provider: AWS

Compute Region: US-East

Carbon Emitted: Estimated 2.3 kg CO₂eq

Technical Specifications
Model Architecture
TinyLlama 1.1B parameters

LoRA adapters on attention layers

FAISS vector store for retrieval

Software
PyTorch

Transformers

PEFT

FAISS

Sentence Transformers

Citation  
BibTeX:
```
@misc{dreaminterpretationai2025,
  author = {Farah Asaad Kayani},
  title = {Dream Interpretation AI using RAG + LoRA Fine-Tuning},
  year = {2025},
  howpublished = {\url{[https://github.com/farahkayani/Dream-Interpretation-AI}}
}
```
