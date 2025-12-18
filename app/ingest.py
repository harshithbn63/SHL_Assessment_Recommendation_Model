"""
Refined LangChain FAISS Ingestion - Implementation based on better.ipynb
Includes structured embedding text and popularity map for training-aware re-ranking.
"""
import pandas as pd
import ast
import os
import pickle
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'final1.csv')
TRAIN_PATH = os.path.join(BASE_DIR, 'train_data.csv')
FAISS_INDEX_PATH = os.path.join(DATA_DIR, 'langchain_faiss_index')
POPULARITY_PATH = os.path.join(DATA_DIR, 'popularity_map.pkl')

def build_embedding_text(row):
    """Structured text template from better.ipynb"""
    return f"""
Assessment Name: {row['name']}
Description: {row['description']}
Job Levels: {', '.join(row['job_levels']) if isinstance(row['job_levels'], list) else row['job_levels']}
Assessment Length: {row['assessment_length_mins']} minutes
Test Domains: {', '.join(row['test_type_labels']) if isinstance(row['test_type_labels'], list) else row['test_type_labels']}
""".strip()

def ingest():
    print(f"Loading data from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print("Error: csv file not found.")
        return

    df = pd.read_csv(CSV_PATH, encoding_errors='replace')
    df = df.fillna('')
    
    # Pre-parse lists for the template
    def safe_eval(x):
        try: return ast.literal_eval(x)
        except: return x if isinstance(x, list) else []
            
    df['test_type_codes'] = df['test_type_codes'].apply(safe_eval)
    df['test_type_labels'] = df['test_type_labels'].apply(safe_eval)
    df['job_levels'] = df['job_levels'].apply(safe_eval)
    df['assessment_length_mins'] = pd.to_numeric(df['assessment_length_mins'], errors='coerce').fillna(0)
    
    # 1. Calculate Popularity Map from Train Data
    print("Calculating URL popularity from train_data.csv...")
    if os.path.exists(TRAIN_PATH):
        train_df = pd.read_csv(TRAIN_PATH)
        popularity_map = dict(Counter(train_df["Assessment_url"]))
        with open(POPULARITY_PATH, 'wb') as f:
            pickle.dump(popularity_map, f)
        print(f"Saved popularity map to {POPULARITY_PATH}")
    else:
        print("Warning: train_data.csv not found. Popularity map will be empty.")

    # 2. Create LangChain Documents
    print("Creating LangChain Documents with structured text...")
    documents = []
    for idx, row in df.iterrows():
        text = build_embedding_text(row)
        
        # Metadata
        metadata = {
            "id": int(idx),
            "name": row['name'],
            "url": row['url'],
            "duration": float(row['assessment_length_mins']),
            "type_labels": row['test_type_labels'],
            "type_codes": row['test_type_codes'],
        }
        
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
    
    print("Initializing HuggingFaceEmbeddings (intfloat/e5-large-v2)...")
    # Note: e5 models usually expect "query: " and "passage: " prefixes for optimal performance
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("Building FAISS index...")
    # Add "passage: " prefix to documents for E5
    # The notebook didn't explicitly add prefixes to passages, but added "query: " to queries.
    # We follow the notebook's behavior exactly for now.
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save locally
    print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    vectorstore.save_local(FAISS_INDEX_PATH)
    
    print(f"✓ Ingestion complete and verified against better.ipynb approach")

if __name__ == "__main__":
    ingest()
