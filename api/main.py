from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import pickle
import os
from typing import List, Dict
import time

app = FastAPI()

# AI Proxy configuration
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/"
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDYwMDNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.EAaEU4BoV1naEeUnwm0gOVPQ08KI7tbKls2O3PKuOjI"

# Define request model
class Query(BaseModel):
    question: str
    image: str | None = None

# Load post metadata
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data")
index_path = os.path.join(data_dir, "post_index.pkl")

try:
    with open(index_path, "rb") as f:
        index_data = pickle.load(f)
    metadata = index_data["metadata"]
except Exception as e:
    print(f"Error loading index: {e}")
    raise Exception("Failed to load post index")

# Function to get embeddings with batching and retries
def get_embeddings(texts: List[str], batch_size: int = 10, retries: int = 3) -> List[List[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{AIPROXY_URL}v1/embeddings",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {AIPROXY_TOKEN}"
                    },
                    json={
                        "model": "text-embedding-3-small",
                        "input": batch
                    },
                    timeout=10
                )
                response.raise_for_status()
                batch_embeddings = [item["embedding"] for item in response.json()["data"]]
                embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                print(f"Error getting embeddings for batch {i//batch_size + 1}, attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    print(f"Failed to get embeddings for batch. Using zeros.")
                    embeddings.extend([[0] * 1536 for _ in batch])
                time.sleep(2 ** attempt)  # Exponential backoff
    return embeddings

# Function to get chat completion
def get_chat_completion(question: str, context: str) -> str:
    try:
        response = requests.post(
            f"{AIPROXY_URL}v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AIPROXY_TOKEN}"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a teaching assistant for a data science course. Provide a concise, accurate answer to the student's question, using the provided context from Discourse posts if relevant."},
                    {"role": "user", "content": f"Question: {question}\nContext: {context}"}
                ],
                "max_tokens": 200
            },
            timeout=15
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error getting chat completion: {e}")
        return "Sorry, I couldn't generate an answer. Please refer to the linked Discourse posts."

# Precompute embeddings for posts
embedding_cache_path = os.path.join(data_dir, "post_embeddings.pkl")
if os.path.exists(embedding_cache_path):
    try:
        with open(embedding_cache_path, "rb") as f:
            post_embeddings = pickle.load(f)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        post_embeddings = None
else:
    print("Generating post embeddings...")
    post_texts = [post["text"] for post in metadata]
    post_embeddings = get_embeddings(post_texts)
    try:
        with open(embedding_cache_path, "wb") as f:
            pickle.dump(post_embeddings, f)
        print(f"Saved embeddings to {embedding_cache_path}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

if not post_embeddings:
    print("Warning: No valid post embeddings available. Search may be limited.")

@app.post("/api/")
async def answer_question(query: Query):
    try:
        # Extract question
        question = query.question
        
        # Get question embedding
        question_embedding = get_embeddings([question])[0]
        if not question_embedding or all(v == 0 for v in question_embedding):
            raise Exception("Failed to get question embedding")
        
        # Compute similarities
        similarities = [
            np.dot(question_embedding, post_emb) / (np.linalg.norm(question_embedding) * np.linalg.norm(post_emb))
            if np.linalg.norm(post_emb) > 0 else 0
            for post_emb in post_embeddings
        ]
        top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5 posts
        
        # Collect relevant posts
        links = []
        context = ""
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Relevance threshold
                post = metadata[idx]
                links.append({
                    "url": post["url"],
                    "text": post["text"][:200] + "..." if len(post["text"]) > 200 else post["text"]
                })
                context += f"- {post['text'][:500]}\n"
        
        # Generate answer
        answer = get_chat_completion(question, context)
        
        # Return response
        return {
            "answer": answer,
            "links": links
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)