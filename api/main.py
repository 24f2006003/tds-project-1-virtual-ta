from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import re
import base64
from openai import OpenAI

# Only load dotenv in development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
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

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

def resolve_path(relative_path: str) -> str:
    """Resolve path for both local and Vercel environments"""
    if os.getenv("VERCEL"):
        return os.path.join("/var/task", relative_path)
    else:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(current_dir, relative_path)

# Load data files
def load_data():
    """Load course content and discourse posts from JSON files"""
    try:
        course_content_path = resolve_path("data/course_content.json")
        discourse_posts_path = resolve_path("data/discourse_posts.json")
        
        with open(course_content_path, "r", encoding="utf-8") as f:
            course_content = json.load(f)
        
        with open(discourse_posts_path, "r", encoding="utf-8") as f:
            discourse_posts = json.load(f)
        
        print(f"Loaded {len(course_content)} course content entries and {len(discourse_posts)} discourse posts")
        return course_content, discourse_posts
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return [], []

# Load data at startup
course_content, discourse_posts = load_data()

def get_system_prompt():
    """System prompt for the Virtual TA"""
    return (
        "You are a Virtual Teaching Assistant for the Tools in Data Science course at IIT Madras. "
        "Answer questions using ONLY the provided context. Be concise and direct.\n"
        "Rules:\n"
        "1. Give short, specific answers from the context\n"
        "2. If information is not in context, say 'Information not available in course materials'\n"
        "3. Don't add explanations beyond what's in the context\n"
        "4. For dates/deadlines, only state what's explicitly mentioned\n"
        "5. Keep responses under 100 words unless detailed explanation is needed\n"
    )

def extract_text_content(item: Any) -> str:
    """Recursively extract all text content from nested JSON structures"""
    if isinstance(item, str):
        return item.strip()
    elif isinstance(item, dict):
        texts = []
        for key, value in item.items():
            if key.lower() not in ['id', 'index', 'timestamp']:  # Skip metadata fields
                text = extract_text_content(value)
                if text:
                    texts.append(text)
        return " ".join(texts)
    elif isinstance(item, list):
        texts = []
        for subitem in item:
            text = extract_text_content(subitem)
            if text:
                texts.append(text)
        return " ".join(texts)
    else:
        return str(item) if item is not None else ""

def simple_search(question: str) -> List[Dict]:
    """Simple text search that actually finds content"""
    question_lower = question.lower()
    results = []
    
    print(f"Searching for: '{question}'")
    print(f"Available discourse posts: {len(discourse_posts)}")
    print(f"Available course content: {len(course_content)}")
    
    # Search discourse posts - just check if ANY word from question appears in content
    for idx, post in enumerate(discourse_posts):
        try:
            content_text = extract_text_content(post)
            if not content_text:
                continue
            
            content_lower = content_text.lower()
            
            # Very simple matching - if any significant word appears, include it
            question_words = [w for w in question_lower.split() if len(w) > 2]
            
            found_match = False
            for word in question_words:
                if word in content_lower:
                    found_match = True
                    break
            
            if found_match:
                results.append({
                    'content': content_text,
                    'source': 'discourse',
                    'title': post.get('title', 'Discussion Post'),
                    'url': post.get('url', '#'),
                    'raw_data': post
                })
                print(f"Found discourse match: {post.get('title', 'No title')[:50]}")
        
        except Exception as e:
            print(f"Error processing discourse post {idx}: {e}")
            continue
    
    # Search course content
    for idx, content_item in enumerate(course_content):
        try:
            content_text = extract_text_content(content_item)
            if not content_text:
                continue
            
            content_lower = content_text.lower()
            
            # Very simple matching
            question_words = [w for w in question_lower.split() if len(w) > 2]
            
            found_match = False
            for word in question_words:
                if word in content_lower:
                    found_match = True
                    break
            
            if found_match:
                results.append({
                    'content': content_text,
                    'source': 'course',
                    'title': content_item.get('title', 'Course Content'),
                    'url': content_item.get('url', '#'),
                    'raw_data': content_item
                })
                print(f"Found course match: {content_item.get('title', 'No title')[:50]}")
        
        except Exception as e:
            print(f"Error processing course content {idx}: {e}")
            continue
    
    print(f"Total matches found: {len(results)}")
    return results

def build_context(question: str) -> tuple[str, List[Dict]]:
    """Build context from search results"""
    search_results = simple_search(question)
    
    if not search_results:
        print("No search results found")
        return "", []
    
    context_parts = []
    links = []
    
    # Take first 5 results to avoid token limits
    for result in search_results[:5]:
        content = result['content']
        title = result.get('title', '')
        
        # Format the content block
        header = f"[{result['source'].upper()}]"
        if title:
            header += f" {title}"
        
        content_block = f"{header}\n{content}\n"
        context_parts.append(content_block)
        
        # Add link
        links.append({
            'url': result.get('url', '#'),
            'text': title or 'Related Content'
        })
    
    context = "\n---\n".join(context_parts)
    print(f"Built context with {len(context)} characters from {len(context_parts)} sources")
    
    return context, links

class Query(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
async def answer_question(query: Query):
    """Main endpoint to answer questions"""
    try:
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        print(f"Processing question: {query.question}")
        
        if not course_content and not discourse_posts:
            return {
                "answer": "No course materials are currently available.",
                "links": [{'url': '#', 'text': 'Course Materials'}]
            }
        
        # Build context from search results
        context, links = build_context(query.question)
        
        if not context.strip():
            return {
                "answer": "Information not available in course materials",
                "links": [{'url': '#course-materials', 'text': 'Course Materials Reference'}]
            }
        
        # Prepare messages for OpenAI
        messages = [
            {
                "role": "system", 
                "content": get_system_prompt()
            },
            {
                "role": "user",
                "content": f"""Context: {context}

Question: {query.question}

Answer briefly and directly from the context only."""
            }
        ]
        
        # Call OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=200
            )
            
            if response.choices and response.choices[0].message.content:
                answer = response.choices[0].message.content.strip()
                
                # Ensure we always return links
                if not links:
                    links = [{'url': '#course-materials', 'text': 'Related Course Materials'}]
                
                return {
                    "answer": answer,
                    "links": links[:5]
                }
            else:
                return {
                    "answer": "I couldn't generate a response. Please try rephrasing your question.",
                    "links": links if links else [{'url': '#', 'text': 'Course Materials'}]
                }
                
        except Exception as api_error:
            print(f"OpenAI API error: {str(api_error)}")
            return {
                "answer": f"API Error: {str(api_error)}",
                "links": links if links else [{'url': '#', 'text': 'Course Materials'}]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in answer_question: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "links": [{'url': '#', 'text': 'Course Materials'}]
        }

# Debug endpoint to check what's in the data
@app.get("/debug/data")
async def debug_data():
    """Debug endpoint to see what data is loaded"""
    return {
        "course_content_count": len(course_content),
        "discourse_posts_count": len(discourse_posts),
        "sample_course_content": course_content[:2] if course_content else [],
        "sample_discourse_posts": discourse_posts[:2] if discourse_posts else []
    }

# Debug endpoint to test search
@app.get("/debug/search/{question}")
async def debug_search(question: str):
    """Debug endpoint to test search functionality"""
    results = simple_search(question)
    return {
        "question": question,
        "results_count": len(results),
        "results": results[:3]  # First 3 results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)