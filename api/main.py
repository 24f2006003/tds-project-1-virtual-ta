from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")
if not OPENAI_BASE_URL:
    raise ValueError("OPENAI_BASE_URL environment variable not set")

# Initialize OpenAI client - it will automatically use OPENAI_API_KEY and OPENAI_BASE_URL from env
client = OpenAI()

# Load data files
def load_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "data")
        
        with open(os.path.join(data_dir, "course_content.json"), "r", encoding="utf-8") as f:
            course_content = json.load(f)
        with open(os.path.join(data_dir, "discourse_posts.json"), "r", encoding="utf-8") as f:
            discourse_posts = json.load(f)
        return course_content, discourse_posts
    except Exception as e:
        print(f"Error loading data files: {e}")
        return [], []

course_content, discourse_posts = load_data()

def find_relevant_context(question: str, max_results: int = 3) -> Tuple[str, List[Dict]]:
    """Find relevant context from course content and discourse posts."""
    relevant_posts = []
    
    # Simple keyword matching - in a production environment, you'd want to use
    # embeddings and semantic search for better results
    keywords = question.lower().split()
    
    # Search discourse posts
    post_scores = []
    for idx, post in enumerate(discourse_posts):
        score = sum(1 for keyword in keywords if keyword in post["text"].lower())
        if score > 0:
            # Include the index to ensure stable sorting
            post_scores.append((score, idx, post))
    
    # Get top matching posts
    sorted_posts = sorted(post_scores, key=lambda x: (-x[0], x[1]))  # Sort by score desc, then index asc
    relevant_posts = [post for _, _, post in sorted_posts[:max_results]]
    
    # Get course content
    course_context = ""
    if course_content and len(course_content) > 0:
        course_context = course_content[0].get("text", "")
    
    # Combine context
    contexts = []
    if course_context:
        # Look for specific sections about prerequisites or requirements
        lower_context = course_context.lower()
        relevant_sections = [
            ("Programming skills are a pre-requisite", "This course teaches you tools"),
            ("you need a good understanding", "But isn't this a data science course"),
            ("This course is quite hard", "Programming skills are a pre-requisite"),
        ]
        
        for start_phrase, end_phrase in relevant_sections:
            start_idx = lower_context.find(start_phrase.lower())
            if start_idx != -1:
                end_idx = lower_context.find(end_phrase.lower(), start_idx)
                if end_idx != -1:
                    contexts.append(course_context[start_idx:end_idx].strip())
    
    # Add relevant discourse posts about prerequisites
    for post in relevant_posts:
        if any(keyword in post["text"].lower() for keyword in ["prerequisite", "require", "need to know", "before taking"]):
            contexts.append(post["text"][:500])
    
    combined_context = "\n\n".join(contexts)
    
    return combined_context, relevant_posts

class Query(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
async def answer_question(query: Query):
    try:
        # Get relevant context and posts
        context, relevant_posts = find_relevant_context(query.question)
        
        # Prepare conversation with context
        messages = [
            {"role": "system", "content": """You are a teaching assistant for the Tools in Data Science course at IIT Madras. 
            Your answers should be based ONLY on the course content and student discussions provided.
            When discussing prerequisites, focus on the specific skills and knowledge students need BEFORE taking the course,
            not the topics that will be covered during the course.
            Be direct and specific, citing information from the course materials when possible.
            If the information isn't clearly stated in the provided context, say so."""},
            {"role": "user", "content": f"""Course Context:
            {context}
            
            Question: {query.question}
            
            Important: Focus only on what students need to know BEFORE taking the course, not what they will learn during it.
            Base your answer only on the course content provided above."""}
        ]
        
        if query.image:
            messages.append({"role": "user", "content": f"I've also attached an image with this question: {query.image}"})
        
        # Get answer from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,  # Lower temperature for more focused answers
            max_tokens=300,
        )
        
        # Extract relevant links
        links = [post["url"] for post in relevant_posts if any(keyword in post["text"].lower() 
                for keyword in ["prerequisite", "require", "need to know", "before taking"])]
        
        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "links": links
        }
    except Exception as e:
        print(f"Error: {str(e)}")  # Add logging
        raise HTTPException(status_code=500, detail=str(e))