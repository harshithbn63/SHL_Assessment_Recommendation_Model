from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .recommender import AssessmentRecommender
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="SHL Assessment Recommendation System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load recommender (FAISS + models)
rec = AssessmentRecommender()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    return rec.recommend(request.query)

# Serve frontend at /ui
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(base_dir, "static")

if os.path.exists(static_dir):
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="static")
