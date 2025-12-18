# SHL Assessment Recommendation System

A state-of-the-art **LLM-enhanced RAG system** designed to provide intelligent SHL assessment recommendations. This system leverages **Google Gemini APIs** for high-performance semantic retrieval and query understanding, making it lightweight and deployment-ready.

## 🚀 Key Features

- **Advanced Semantic Search**: Powered by Google's `text-embedding-004` (768-dimensional embeddings) for superior retrieval accuracy.
- **Gemini Intelligence**: Uses **Gemini-2.0-Flash** for structured hiring intent extraction (skills, job levels, and time constraints).
- **Training-Aware Re-ranking**: Incorporates historical popularity bias from `train_data.csv` to prioritize proven assessments.
- **Round-Robin Diversity**: Ensure recommendations cover diverse assessment domains.
- **Automated URL Scraping**: Directly extracts job requirements from live job posting URLs via BeautifulSoup4.

## 🛠️ Technology Stack

- **Framework**: [LangChain](https://www.langchain.com/)
- **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss)
- **LLM**: Google [Gemini-2.0-Flash](https://ai.google.dev/) (via API)
- **Embeddings**: Google [Text-Embedding-004](https://ai.google.dev/)
- **Backend**: FastAPI (Python 3.11+)
- **Frontend**: Vanilla JS with Glassmorphism UI

## 📋 Setup & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```
*Note: Create a `.env` file with your `GOOGLE_API_KEY` to enable the models.*

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
