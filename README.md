# SHL Assessment Recommendation System

A state-of-the-art **LLM-enhanced RAG system** designed to provide intelligent SHL assessment recommendations from job descriptions and queries. This system leverages the **E5-Large** embedding model and **LangChain** for robust semantic retrieval.

## 🚀 Key Features

- **Advanced Semantic Search**: Powered by `intfloat/e5-large-v2` (1024-dimensional embeddings) for superior retrieval accuracy.
- **Local LLM Intelligence**: Uses Google **Flan-T5-small** for structured hiring intent extraction (skills, job levels, and time constraints).
- **Training-Aware Re-ranking**: Incorporates historical popularity bias from `train_data.csv` to prioritize proven assessments.
- **Round-Robin Diversity**: A balanced selection algorithm that ensures recommendations cover diverse assessment domains (Ability, Skills, Personality).
- **Automated URL Scraping**: Directly extracts job requirements from live job posting URLs via BeautifulSoup4.

## 🛠️ Technology Stack

- **Framework**: [LangChain](https://www.langchain.com/) (VectorStore, Documents, Embeddings)
- **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss)
- **LLM**: Google [Flan-T5-small](https://huggingface.co/google/flan-t5-small) (Running locally via Transformers)
- **Embeddings**: [E5-Large-v2](https://huggingface.co/intfloat/e5-large-v2)
- **Backend**: FastAPI (Python 3.11+)
- **Frontend**: Vanilla JS with Glassmorphism UI

## 📋 Setup & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```
*Note: The first run will automatically download the local LLM (~308MB) and Embedding models.*

### 2. Ingestion (Optional)
The FAISS index is pre-built in `data/langchain_faiss_index`. To rebuild it from `final1.csv`:
```bash
python app/ingest.py
```

### 3. Start the Server
```bash
python -m app.main
```
Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

## 📊 Evaluation & Performance

The system is validated against the provided `train_data.csv` using the **Mean Recall@10** metric.

- **Submission File**: `data/submission.csv` (generated for `test_data.csv`)
- **Run Evaluation**:
  ```bash
  python scripts/evaluate.py
  ```

---
*Developed as part of the SHL Recruitment Recommendation Challenge.*
