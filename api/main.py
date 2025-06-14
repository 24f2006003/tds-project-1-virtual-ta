from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
import os
import json
from openai import OpenAI

# Only load dotenv in development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # We're probably on Vercel, which doesn't need dotenv
    pass

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
    print("Warning: OPENAI_API_KEY environment variable not set")
if not OPENAI_BASE_URL:
    print("Warning: OPENAI_BASE_URL environment variable not set")

# Initialize OpenAI client - it will automatically use OPENAI_API_KEY and OPENAI_BASE_URL from env
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

def resolve_path(relative_path: str) -> str:
    """Resolve path for both local and Vercel environments"""
    if os.getenv("VERCEL"):
        # On Vercel, use /var/task
        return os.path.join("/var/task", relative_path)
    else:
        # Local development - use current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, relative_path)

# Load data files
def load_data():
    try:
        data_paths = [
            resolve_path("data/course_content.json"),
            resolve_path("data/discourse_posts.json"),
        ]
        print(f"Attempting to load data from: {data_paths}")
        
        with open(data_paths[0], "r", encoding="utf-8") as f:
            course_content = json.load(f)
        with open(data_paths[1], "r", encoding="utf-8") as f:
            discourse_posts = json.load(f)
        
        print(f"Successfully loaded data files. Course content entries: {len(course_content)}, Discourse posts: {len(discourse_posts)}")
        return course_content, discourse_posts
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        # Return minimal data structure
        return [{"text": "Tools in Data Science is a practical diploma level data science course at IIT Madras."}], []

# Load data at startup
course_content, discourse_posts = load_data()

def find_relevant_context(question: str, max_results: int = 3) -> Tuple[str, List[Dict]]:
    """Find relevant context from course content and discourse posts."""
    try:
        relevant_posts = []
        
        # Simple keyword matching - in a production environment, you'd want to use
        # embeddings and semantic search for better results
        keywords = question.lower().split()
        
        # Search discourse posts
        post_scores = []
        for idx, post in enumerate(discourse_posts):
            post_text = post.get("text", "") if isinstance(post, dict) else str(post)
            score = sum(1 for keyword in keywords if keyword in post_text.lower())
            if score > 0:
                # Include the index to ensure stable sorting
                post_scores.append((score, idx, post))
        
        # Get top matching posts
        sorted_posts = sorted(post_scores, key=lambda x: (-x[0], x[1]))  # Sort by score desc, then index asc
        relevant_posts = [post for _, _, post in sorted_posts[:max_results]]
        
        # Get course content
        course_context = ""
        if course_content and len(course_content) > 0:
            # Get the overview text from course content
            course_context = course_content[0].get("text", "")[:2000]  # Take first 2000 chars of course content
        
        # Combine context
        contexts = []
        if course_context:
            contexts.append(course_context)
        for post in relevant_posts:
            post_text = post.get("text", "") if isinstance(post, dict) else str(post)
            contexts.append(post_text[:500])  # Take first 500 chars of each post
        
        combined_context = "\n\n".join(contexts)
        return combined_context, relevant_posts
    except Exception as e:
        print(f"Error in find_relevant_context: {str(e)}")
        return "", []

class Query(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
async def answer_question(query: Query):
    try:
        # Log incoming request
        print(f"Received question: {query.question}")
        
        # Get relevant context and posts
        context, relevant_posts = find_relevant_context(query.question)
        
        # Log context found
        print(f"Found {len(relevant_posts)} relevant posts")
        print(f"Context length: {len(context)} characters")
        # Prepare conversation with context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a teaching assistant for the Tools in Data Science course at IIT Madras. "
                    "Follow these guidelines strictly:\n\n"
                    "1. Answer ONLY what is asked, using ONLY the provided course content and discussions.\n"
                    "2. DO NOT provide any extra information, background, or general knowledge beyond what is strictly required to answer the question.\n"
                    "3. You MUST NOT use any external knowledge, general information, or sources outside the provided context.\n"
                    "4. If the context does not contain enough information to answer accurately, clearly state that you cannot answer based on the provided materials.\n"
                    "5. For future dates or schedules (exams, deadlines, etc.), clearly state that this information "
                    "will be announced later unless it's explicitly mentioned in the context.\n"
                    "6. For technical choices (Docker vs Podman, etc.), give advice ONLY if the context provides it. If the context does not mention a tool, do not recommend or explain it.\n"
                    "7. For grading or dashboard display questions, answer ONLY if the context explicitly describes the behavior.\n"
                    "8. If you're not completely sure about something, explicitly say so.\n"
                    "9. Keep answers as brief and direct as possible.\n"
                    "10. Never assume or infer details not present in the provided context.\n"
                )
            },
            {
                "role": "user",
                "content": f"""Context from course materials and discussions:
{context}

Question: {query.question}"""
            }
        ]
        if query.image:
            messages.append({
                "role": "user",
                "content": f"I've also attached an image with this question: {query.image}"
            })
        
        try:
            # Get answer from OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini as it's supported by the AI proxy
                messages=messages,
                temperature=0.05,  # Set to 0.05 for slight randomness while maintaining focus
                max_tokens=500  # Set to 500 for concise, focused answers without unnecessary verbosity
            )
            # Check for errors in the response
            if response.choices[0].message.content:
                # Extract relevant links
                links = []
                for post in relevant_posts:
                    if "url" in post:
                        links.append({
                            "url": post["url"],
                            "text": post.get("title", "Related discussion")
                        })

                answer = response.choices[0].message.content
                print(f"Successfully generated answer of length: {len(answer)}")

                return {
                    "answer": answer,
                    "links": links
                }
            else:
                # Fallback if no content
                return {
                    "answer": "I couldn't generate a response based on the available course materials.",
                    "links": []
                }
            
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            # Return fallback response instead of error
            return {
                "answer": "I'm having trouble accessing the AI service right now. Please try again later.",
                "links": []
            }
            
    except Exception as e:
        print(f"Error in answer_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))