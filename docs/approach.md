# SHL Assessment Recommendation System - Technical Approach

## 1. Executive Summary
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to solve the problem of matching job descriptions to technical and behavioral assessments. By combining **E5-Large embeddings** for search and **Flan-T5** for query understanding, the system provides high-precision, context-aware recommendations that outperform simple keyword matching.

---

## 2. System Architecture

The recommendation engine follows a multi-stage pipeline:

### A. Intent Extraction (Local LLM)
- **Model**: Google `flan-t5-small`.
- **Function**: Parses raw queries into structured JSON containing:
  - `skills`: Specific technical or domain requirements.
  - `job_level`: Seniority context (Entry, Mid, Senior).
  - `max_duration`: Time constraints (e.g., "fast assessment").
  - `role_family`: General domain (Engineering, Sales, etc.).

### B. Embedding & Retrieval (LangChain + FAISS)
- **Model**: `intfloat/e5-large-v2`.
- **Process**: 
  - Each assessment is transformed into a structured document.
  - Documents are indexed in a **FAISS** vector store via **LangChain**.
  - Queries are augmented with a `query: ` prefix (optimizing E5 performance) and matched against the index using cosine similarity.

### C. Training-Aware Re-ranking
- **Popularity Bias**: The system extracts a "popularity map" from `train_data.csv`.
- **Scoring Logic**: Candidates are re-ranked using an alpha-weighted combination of semantic distance and historical popularity:
  `final_score = semantic_distance - (0.2 * popularity_boost)`
  This prioritizes assessments that are statistically most relevant to the provided training ground truth.

### D. Diversity Balancing (Round-Robin)
- To avoid over-suggesting a single type of test (e.g., only math tests), the system categorizes retrieved docs into buckets (Ability, Skills, Personality).
- Final results are selected using a round-robin strategy to ensure the top 10 results provide a comprehensive evaluation of the candidate.

---

## 3. Compliance & Standards
- **Frameworks**: 100% compliant with **LangChain** and **FAISS** requirements.
- **Privacy**: Uses **local models** exclusively; no external API calls (OpenAI, Anthropic, etc.) are made during recommendation.
- **Reproducibility**: Includes a standalone `evaluate.py` script for performance verification.

---

## 4. Key Performance Improvements
| Feature | Impact |
| :--- | :--- |
| **E5-Large v2** | 1024D embeddings capture nuance (e.g., "Python for Data" vs "Python for Web"). |
| **Popularity Bias** | Aligns recommendations with historical HR preferences in the training data. |
| **Balanced Selection** | Ensures a "whole-person" assessment recommendation (Ability + Personality + Skills). |
| **JSON LLM Parsing** | Replaces brittle regex with robust natural language understanding. |
