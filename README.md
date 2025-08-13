Dream Interpretation AI 🌙  
A Retrieval-Augmented Generation (RAG) + LoRA Fine-Tuned Model for Context-Aware Dream Analysis

📌 Overview:  

The Dream Interpretation AI is a domain-specific NLP system designed to provide accurate, culturally aware, and personalized interpretations of user-submitted dreams. It combines:  

Retrieval-Augmented Generation (RAG) for grounding interpretations in similar past dream cases.  

Low-Rank Adaptation (LoRA) fine-tuning to adapt a base language model for the dream interpretation domain without full retraining.  

The system retrieves semantically relevant dream narratives from a FAISS-powered vector database, formats them as contextual input, and passes them to a LoRA fine-tuned TinyLlama model for generating insightful interpretations.

🚀 Features: 
Context-Aware Responses — Incorporates real dream examples into interpretations.  

Efficient Fine-Tuning — Uses LoRA for domain adaptation with minimal compute cost.  

Semantic Search — FAISS + Sentence Transformers for accurate retrieval.  

Culturally Attuned — Trained on curated dream interpretation datasets.  
  
🏗 Architecture  
User Input — User describes their dream.  

Document Retrieval — FAISS + Sentence Transformers retrieve similar dreams.

Context Construction — Retrieved cases formatted for LLM input.

LLM Generation — LoRA fine-tuned TinyLlama generates the interpretation.

Response Delivery — Output is presented to the user.

📂 Project Structure
```
dream-interpretation-ai/
│
├── data/                          # Dream corpus and FAISS index files
│   ├── corpus.jsonl
│   ├── faiss.index
│   └── meta.pkl
│
├── build_index.py                  # Builds FAISS index from corpus
├── rag_query.py                    # Runs the interactive RAG-powered interpretation
├── prepare_lora_dataset.py         # Converts corpus into LoRA fine-tuning format
├── lora_train.py                   # Fine-tunes the model with LoRA
│
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```


⚙️ Installation
1️⃣ Clone the Repository

git clone https://github.com/farahkayani/Dream-Interpretation-AI/tree/main  
cd dream-interpretation-ai

2️⃣ Install Dependencies

pip install -r requirements.txt

📊 Usage
Steps:
1. Build the FAISS Index
   python build_index.py

2. Run the AI Interpreter
   python rag_query.py

3. Prepare Dataset for LoRA Training
   python prepare_lora_dataset.py

4. Train with LoRA
   python lora_train.py
   

📦 Requirements  
Python 3.8+

PyTorch

sentence-transformers

faiss

transformers

peft

datasets

Install all dependencies with:
pip install -r requirements.txt

📝 Example
User Input:
“I was walking through a quiet garden when I suddenly saw a large snake coiled around a tree. Its eyes seemed to follow me, and I felt a mix of fear and fascination as it slithered closer.”

AI Output:
- Related dreams
"Seeing a snake in a dream may symbolize hidden threats, betrayal, or transformation. The snake's behavior and your reaction to it greatly influence its meaning.”
- Interpretation
"The dream's main message is that you may have been confronted with a hidden danger or threat in your life.”


📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

👩‍💻 Author
Farah — AI and NLP Developer
