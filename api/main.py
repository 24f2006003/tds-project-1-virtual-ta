from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
import pickle
import os
import json
from typing import List, Dict
import time

app = FastAPI()
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI configuration
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# Define request model
class Query(BaseModel):
    question: str
    image: str | None = None

# Load post metadata directly from JSON
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "..", "data", "discourse_posts.json")

try:
    with open(json_path, "r", encoding="utf-8") as f:
        posts = json.load(f)
    metadata = [{"url": p["url"], "text": p["text"][:500]} for p in posts[:30]]
except Exception as e:
    print(f"Error reading JSON: {e}")
    # Provide fallback empty data instead of failing
    metadata = []

if not metadata:
    print("Warning: No metadata available. API will return limited results.")

# Function to get embeddings
def get_embeddings(texts: List[str], batch_size: int = 5, retries: int = 3) -> List[List[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(retries):
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                print(f"Error getting embeddings for batch {i//batch_size + 1}, attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    embeddings.extend([[0] * 1536 for _ in batch])
                time.sleep(2 ** attempt)
    return embeddings

# Function to get chat completion
def get_chat_completion(question: str, context: str, image: str | None = None) -> str:
    try:
        messages = [
            {"role": "system", "content": "You are a teaching assistant for a data science course. Provide a concise, accurate answer."},
            {"role": "user", "content": f"Question: {question}\nContext: {context}"}
        ]
        
        # Add image context if provided
        if image:
            messages.append({"role": "user", "content": f"I've also attached an image with this question: {image}"})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting chat completion: {e}")
        return "Sorry, I couldn't generate an answer. Please refer to the linked Discourse posts."

# Generate embeddings at startup
print("Generating embeddings...")
post_texts = [post["text"] for post in metadata]
post_embeddings = get_embeddings(post_texts) if post_texts else []

@app.post("/")
@app.options("/")
async def answer_question(request: Request):
    if request.method == "OPTIONS":
        # CORS preflight request
        return {}
    
    try:
        # Handle raw request body
        body = await request.body()
        try:
            data = json.loads(body.decode("utf-8"))
            if isinstance(data, str):
                data = json.loads(data)  # Handle stringified JSON
            query = Query(**data)
        except json.JSONDecodeError:
            query = Query(**await request.json())  # Fallback to FastAPI JSON parsing
        
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
        top_indices = np.argsort(similarities)[-5:][::-1]
        
        # Collect relevant posts
        links = []
        context = ""
        for idx in top_indices:
            if similarities[idx] > 0.1 and idx < len(metadata):
                post = metadata[idx]
                # Get first sentence or first 100 characters for concise text
                text = post["text"].split(".")[0]
                if len(text) > 100:
                    text = text[:97] + "..."
                links.append({
                    "url": post["url"],
                    "text": text
                })
                context += f"- {post['text'][:500]}\n"
        
        # Generate answer
        answer = get_chat_completion(question, context, query.image)
        
        # Return response
        return {
            "answer": answer,
            "links": links
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)