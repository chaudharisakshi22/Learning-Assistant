# 🎓 Personalized Learning Assistant
**Python · LangChain · HuggingFace · RAG · Streamlit**

---

## What It Does

| Feature | How |
|---|---|
| 📂 Ingest study material | PDFs, TXT files, or any URL |
| 🔍 Semantic search | `sentence-transformers/all-MiniLM-L6-v2` + FAISS |
| 📝 AI quiz generation | `Mistral-7B-Instruct` via HuggingFace Inference API |
| 📊 Performance tracking | JSON-based local tracker per topic |
| 💡 Personalised recommendations | LLM coach using weak-topic data + RAG context |

---

## Project Structure

```
learning_assistant/
│
├── app.py                      ← Streamlit UI (run this)
│
├── modules/
│   ├── document_loader.py      ← PDF / TXT / URL → LangChain Documents
│   ├── vector_store.py         ← HuggingFace Embeddings + FAISS index
│   ├── llm_setup.py            ← HuggingFace Inference API (Mistral-7B)
│   ├── quiz_generator.py       ← RAG-powered MCQ generation + parser
│   ├── performance_tracker.py  ← JSON-based score tracking per topic
│   └── recommender.py          ← AI personalised study plan
│
├── data/
│   ├── vectorstore/            ← FAISS index (auto-created)
│   └── performance.json        ← Quiz scores (auto-created)
│
├── .env.example                ← Copy to .env and add your HF token
└── requirements.txt
```

---

## ⚡ Quick Setup (Step-by-Step for Beginners)

### Step 1 — Clone / download the project
```bash
cd learning_assistant
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Get your FREE HuggingFace token
1. Go to https://huggingface.co/settings/tokens
2. Click **New token** → name it anything → **Read** permission
3. Copy the token (starts with `hf_...`)

### Step 5 — Set up your .env file
```bash
cp .env.example .env
```
Open `.env` and paste your token:
```
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
```

### Step 6 — Run the app
```bash
streamlit run app.py
```
Your browser will open at **http://localhost:8501** 🚀

---

## 🧪 How to Use the App

1. **Upload material** — Use the sidebar to upload a PDF, TXT, or paste a URL
2. **Click "Ingest Material"** — It will embed your documents into the vector store
3. **Go to "Quiz Generator"** tab — Type a topic from your material and generate a quiz
4. **Submit the quiz** — See your score and explanations
5. **Check "My Progress"** — See stats per topic
6. **Get "Recommendations"** — Click the button for a personalised AI study plan

---

## 🔧 Customisation Tips

| What to change | Where |
|---|---|
| Swap LLM model | `LLM_MODEL_ID` in `.env` |
| Change embedding model | `EMBEDDING_MODEL_ID` in `.env` |
| Chunk size for splitting | `CHUNK_SIZE` in `modules/document_loader.py` |
| Number of retrieved chunks | `k` param in `vector_store.get_retriever()` |
| Weak topic threshold | `threshold` in `performance_tracker.get_weak_topics()` |

### Other free models you can try on HuggingFace:
- `google/flan-t5-large` (lighter, faster)
- `HuggingFaceH4/zephyr-7b-beta`
- `tiiuae/falcon-7b-instruct`

---

