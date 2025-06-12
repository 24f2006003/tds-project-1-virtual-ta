from fastapi import FastAPI, Request
from rank_bm25 import BM25Okapi
import json
import base64
import easyocr
import requests
import os

app = FastAPI()

# Load data
with open("../data/discourse_posts.json", "r") as f:
    discourse_posts = json.load(f)
with open("../data/course_content.json", "r") as f:
    course_content = json.load(f)
documents = discourse_posts + course_content

# Create BM25 index
corpus = [doc["text"] for doc in documents]
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Initialize OCR
reader = easyocr.Reader(["en"])

# AI Proxy URL
AI_PROXY_URL = "https://aiproxy.sanand.workers.dev/"

@app.post("/api/")
async def answer_question(request: Request):
    data = await request.json()
    question = data["question"]
    image_base64 = data.get("image")
    
    # Process image if provided
    if image_base64:
        image_data = base64.b64decode(image_base64)
        with open("/tmp/image.png", "wb") as f:
            f.write(image_data)
        ocr_result = reader.readtext("/tmp/image.png")
        ocr_text = " ".join([res[1] for res in ocr_result])
        question += " " + ocr_text
        os.remove("/tmp/image.png")
    
    # Retrieve top 5 documents
    query = question.split()
    scores = bm25.get_scores(query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    retrieved_docs = [documents[i] for i in top_k_indices]
    
    # Create prompt for AI proxy
    prompt = f"Question: {question}\n\nContext:\n"
    for doc in retrieved_docs:
        prompt += f"- {doc['text']}\n"
    prompt += "\nProvide an answer based on the context above."
    
    # Call AI proxy
    response = requests.post(AI_PROXY_URL, json={"prompt": prompt})
    answer = response.json().get("completion", "No answer generated.")
    
    # Collect links
    links = [{"url": doc["url"], "text": "Relevant content"} for doc in retrieved_docs]
    
    return {"answer": answer, "links": links}