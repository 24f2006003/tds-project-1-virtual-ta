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
        "Answer questions using ONLY the provided context from course materials and discourse discussions. "
        "\nGuidelines:\n"
        "1. Use the provided context to answer questions accurately\n"
        "2. Quote specific details from the context when relevant\n"
        "3. If the context doesn't contain the requested information, state that it's not available\n"
        "4. Include links when provided in the context\n"
        "5. Be helpful and direct in your responses\n"
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

def search_content(question: str, max_results: int = 10) -> List[Dict]:
    """Search through all content with flexible matching"""
    question_lower = question.lower()
    question_words = re.findall(r'\b\w+\b', question_lower)
    
    results = []
    
    # Search discourse posts
    for idx, post in enumerate(discourse_posts):
        if not isinstance(post, dict):
            continue
            
        content_text = extract_text_content(post)
        if not content_text:
            continue
            
        content_lower = content_text.lower()
        
        # Calculate match score
        score = 0
        
        # Exact phrase matching
        if len(question_words) >= 2:
            for i in range(len(question_words) - 1):
                phrase = f"{question_words[i]} {question_words[i+1]}"
                if phrase in content_lower:
                    score += 10
        
        # Individual word matching
        for word in question_words:
            if len(word) >= 3 and word in content_lower:
                score += 1
                # Bonus for exact word boundaries
                if re.search(r'\b' + re.escape(word) + r'\b', content_lower):
                    score += 2
        
        # Special patterns
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', content_text):  # Dates
            if any(word in question_lower for word in ['date', 'when', 'schedule', 'exam', 'deadline']):
                score += 5
                
        if 'http' in content_text and 'link' in question_lower:
            score += 3
            
        if score > 0:
            results.append({
                'content': content_text,
                'score': score,
                'source': 'discourse',
                'title': post.get('title', ''),
                'url': post.get('url', ''),
                'type': 'post'
            })
    
    # Search course content
    for idx, content_item in enumerate(course_content):
        if not isinstance(content_item, dict):
            continue
            
        content_text = extract_text_content(content_item)
        if not content_text:
            continue
            
        content_lower = content_text.lower()
        
        # Calculate match score
        score = 0
        
        # Exact phrase matching
        if len(question_words) >= 2:
            for i in range(len(question_words) - 1):
                phrase = f"{question_words[i]} {question_words[i+1]}"
                if phrase in content_lower:
                    score += 10
        
        # Individual word matching
        for word in question_words:
            if len(word) >= 3 and word in content_lower:
                score += 1
                # Bonus for exact word boundaries
                if re.search(r'\b' + re.escape(word) + r'\b', content_lower):
                    score += 2
        
        # Special patterns
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', content_text):  # Dates
            if any(word in question_lower for word in ['date', 'when', 'schedule', 'exam', 'deadline']):
                score += 5
                
        if score > 0:
            results.append({
                'content': content_text,
                'score': score,
                'source': 'course',
                'title': content_item.get('title', ''),
                'type': 'content'
            })
    
    # Sort by score and return top results
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def build_context(question: str, max_chars: int = 6000) -> tuple[str, List[Dict]]:
    """Build context string from search results"""
    search_results = search_content(question)
    
    if not search_results:
        print(f"No search results found for: {question}")
        return "", []
    
    print(f"Found {len(search_results)} results for question: {question}")
    for i, result in enumerate(search_results[:3]):
        print(f"  Result {i+1}: Score {result['score']}, Source: {result['source']}, Title: {result.get('title', 'No title')[:50]}")
    
    context_parts = []
    total_chars = 0
    links = []
    
    for result in search_results:
        content = result['content']
        title = result.get('title', '')
        
        # Format the content block
        header = f"[{result['source'].upper()}]"
        if title:
            header += f" {title}"
        
        content_block = f"{header}\n{content}\n"
        
        # Check if we can fit this content
        if total_chars + len(content_block) > max_chars:
            # Try to fit a truncated version
            remaining = max_chars - total_chars - len(header) - 10
            if remaining > 200:
                truncated = content[:remaining] + "..."
                content_block = f"{header}\n{truncated}\n"
                context_parts.append(content_block)
            break
        
        context_parts.append(content_block)
        total_chars += len(content_block)
        
        # Collect links
        if result.get('url'):
            links.append({
                'url': result['url'],
                'text': title or 'Related Discussion'
            })
    
    context = "\n---\n".join(context_parts)
    print(f"Built context: {len(context)} chars from {len(context_parts)} sources")
    
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
        
        if not course_content and not discourse_posts:
            return {
                "answer": "No course materials are currently available.",
                "links": []
            }
        
        # Build context from search results
        context, links = build_context(query.question)
        
        if not context.strip():
            return {
                "answer": "I couldn't find relevant information in the course materials for your question. Please contact the course instructor for more details.",
                "links": []
            }
        
        # Prepare messages for OpenAI
        messages = [
            {
                "role": "system",
                "content": get_system_prompt()
            },
            {
                "role": "user",
                "content": f"""Context from course materials:

{context}

Student Question: {query.question}

Please answer based on the provided context."""
            }
        ]
        
        # Call OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.1,
                max_tokens=800
            )
            
            if response.choices and response.choices[0].message.content:
                answer = response.choices[0].message.content.strip()
                return {
                    "answer": answer,
                    "links": links[:5]  # Limit links
                }
            else:
                return {
                    "answer": "I couldn't generate a response. Please try rephrasing your question.",
                    "links": []
                }
                
        except Exception as api_error:
            print(f"OpenAI API error: {str(api_error)}")
            return {
                "answer": "I'm having trouble processing your question right now. Please try again later.",
                "links": []
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in answer_question: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)