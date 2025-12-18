import pandas as pd
import numpy as np
import sys
import os

# Add app to path to import recommender
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.recommender import AssessmentRecommender

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'train_data.csv') 
TEST_PATH = os.path.join(BASE_DIR, 'test_data.csv')
DATA_DIR = os.path.join(BASE_DIR, 'data')
SUBMISSION_PATH = os.path.join(DATA_DIR, 'submission.csv')

def evaluate():
    rec = AssessmentRecommender()
    if rec.vectorstore is None:
        print("Recommender not initialized. Run ingest.py first.")
        return

    # Load Train
    try:
        if not os.path.exists(TRAIN_PATH):
             print(f"Train file not found at {TRAIN_PATH}")
        else:
            df_train = pd.read_csv(TRAIN_PATH)
            print(f"Evaluating on {len(df_train)} train queries...")
            recalls = []
            
            grouped = df_train.groupby('Query')['Assessment_url'].apply(set).reset_index()
            
            for _, row in grouped.iterrows():
                query = row['Query']
                truth = row['Assessment_url']
                if not truth:
                    continue
                    
                preds = rec.recommend(query, top_k=10)
                pred_urls = set(p['URL'] for p in preds)
                
                # Recall@10
                hit = len(truth.intersection(pred_urls))
                recall = hit / len(truth)
                recalls.append(recall)
                
            mean_recall = np.mean(recalls) if recalls else 0
            print(f"Mean Recall@10: {mean_recall:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

    # Generate Submission
    try:
        if not os.path.exists(TEST_PATH):
             print(f"Test file not found at {TEST_PATH}")
             return

        test_df = pd.read_csv(TEST_PATH)
        queries = test_df['Query'].tolist()
        
        print(f"Generating predictions for {len(queries)} test queries...")
        
        output_rows = []
        for q in queries:
            if pd.isna(q): continue
            preds = rec.recommend(q, top_k=10)
            for p in preds:
                output_rows.append({
                    "Query": q,
                    "Assessment_url": p['URL']
                })
                
        out_df = pd.DataFrame(output_rows)
        out_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Saved submission.csv to {SUBMISSION_PATH}")
        
    except Exception as e:
        print(f"Error generating submission: {e}")

if __name__ == "__main__":
    evaluate()
