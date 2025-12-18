from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .recommender import AssessmentRecommender
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rec = AssessmentRecommender()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is empty")
    
    results = rec.recommend(request.query)
    return results

# Mount static files
# We need to find the absolute path to static folder relative to this file
# This file is in app/main.py, static is in ../static
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(base_dir, 'static')

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
