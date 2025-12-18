"""
Refined Recommender - Integrating approach from better.ipynb
Features:
- Pure LangChain FAISS retrieval
- Flan-T5 JSON-based query parsing
- Popularity-based re-ranking (Training data awareness)
- Balanced domain selection (Diversity)
- URL scraping support
"""
import os
import re
import ast
import json
import pickle
import warnings
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

warnings.filterwarnings("ignore")

class AssessmentRecommender:
    def __init__(self, index_path=None, popularity_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if index_path is None:
            index_path = os.path.join(base_dir, 'data', 'langchain_faiss_index')
        if popularity_path is None:
            popularity_path = os.path.join(base_dir, 'data', 'popularity_map.pkl')
            
        try:
            print(f"Loading LangChain FAISS index...")
            # Use Google Gemini Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            self.vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            
            # Load Popularity Map
            if os.path.exists(popularity_path):
                with open(popularity_path, 'rb') as f:
                    self.popularity_map = pickle.load(f)
                print(f"Loaded popularity map with {len(self.popularity_map)} entries")
            else:
                self.popularity_map = {}
                print("Warning: popularity map not found")
            
            # Initialize Gemini LLM for JSON parsing
            print("Initializing Gemini LLM (gemini-2.0-flash)...")
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.llm = genai.GenerativeModel("gemini-2.0-flash")
            print("✓ System ready")
            
        except Exception as e:
            print(f"Error initializing recommender: {e}")
            self.vectorstore = None

    def _fetch_url_text(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=' ', strip=True)[:5000]
        except: return ""

    def _parse_query_with_llm(self, query):
        """Prompt template from better.ipynb for structured extraction"""
        prompt = f"""
Extract structured hiring intent from the query below.
Return JSON with keys:
skills (list),
job_level (entry | mid | senior),
max_duration_minutes (number or null),
role_family (string).

Query:
{query}

JSON:
"""
        try:
            response = self.llm.generate_content(prompt)
            result = response.text
            # Match JSON block
            match = re.search(r"\{.*\}", result, re.S)
            if match:
                return json.loads(match.group())
            return {}
        except Exception as e:
            print(f"LLM Error: {e}")
            return {}

    def _rerank_with_train_bias(self, docs_with_scores, alpha=0.15):
        """
        Adjust scores based on training data popularity.
        Notebook uses: final_score = score - alpha * popularity_boost (since distance)
        We'll treat our 'score' as a distance (lower is better).
        """
        reranked = []
        for doc, score in docs_with_scores:
            url = doc.metadata.get("url", "")
            popularity_boost = self.popularity_map.get(url, 0)
            # Subtracting from distance score makes it "closer" (better rank)
            # Notebook uses a simple linear boost
            final_score = score - (alpha * popularity_boost)
            reranked.append((doc, final_score))
        
        # Sort by final score (ascending distance)
        reranked.sort(key=lambda x: x[1])
        return reranked

    def _balanced_selection(self, docs, max_total=10):
        """
        Enforce balanced selection across domains (round-robin style)
        from better.ipynb logic.
        """
        buckets = defaultdict(list)
        for doc in docs:
            labels = doc.metadata.get("type_labels", [])
            primary = labels[0] if labels and isinstance(labels, list) else "General"
            buckets[primary].append(doc)

        final = []
        while len(final) < max_total:
            added = False
            for domain in list(buckets.keys()):
                if buckets[domain]:
                    final.append(buckets[domain].pop(0))
                    added = True
                    if len(final) == max_total: break
            if not added: break
        return final

    def recommend(self, query, top_k=10):
        if self.vectorstore is None: return []

        # 1. URL Scraping Check
        url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
        urls = url_pattern.findall(query)
        search_text = query
        if urls:
            fetched = self._fetch_url_text(urls[0])
            if fetched:
                search_text = f"{query} {fetched}" if len(query.strip()) != len(urls[0]) else fetched

        # 2. LLM Parsing
        parsed = self._parse_query_with_llm(search_text)
        print(f"LLM Parsing result: {parsed}")

        # 3. Retrieve Candidate Pool
        docs_with_scores = self.vectorstore.similarity_search_with_score(search_text, k=50)

        # 4. Popularity-Aware Re-ranking
        reranked = self._rerank_with_train_bias(docs_with_scores, alpha=0.2)

        # 5. Metadata Filtering
        max_dur = parsed.get("max_duration_minutes")
        filtered_docs = []
        for doc, score in reranked:
            if max_dur and doc.metadata.get("duration", 0) > max_dur + 5:
                continue
            filtered_docs.append(doc)

        # 6. Balanced Selection (Diversity)
        final_selection = self._balanced_selection(filtered_docs, max_total=top_k)

        # 7. Format
        formatted = []
        for doc in final_selection:
            formatted.append({
                "Assessment Name": doc.metadata["name"],
                "URL": doc.metadata["url"],
                "Score": 1.0, # Placeholder for UI if needed
                "Type": doc.metadata.get("type_labels", []),
                "Duration": doc.metadata.get("duration", 0)
            })
        return formatted

if __name__ == "__main__":
    rec = AssessmentRecommender()
    print(rec.recommend("I need a Java developer with leadership skills, within 45 mins"))
