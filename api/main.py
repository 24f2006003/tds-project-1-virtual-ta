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

def search_content(question: str, max_results: int = 15) -> List[Dict]:
    """Search through all content with much more flexible matching"""
    question_lower = question.lower()
    question_words = re.findall(r'\b\w+\b', question_lower)
    
    # Remove common stop words but keep important ones
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be'}
    question_words = [w for w in question_words if w not in stop_words or len(w) >= 4]
    
    results = []
    
    # Search discourse posts first (higher priority)
    for idx, post in enumerate(discourse_posts):
        if not isinstance(post, dict):
            continue
            
        content_text = extract_text_content(post)
        if not content_text:
            continue
            
        content_lower = content_text.lower()
        
        # Calculate match score with more generous scoring
        score = 0
        
        # Exact phrase matching (higher weight)
        if len(question_words) >= 2:
            for i in range(len(question_words) - 1):
                phrase = f"{question_words[i]} {question_words[i+1]}"
                if phrase in content_lower:
                    score += 20  # High weight for phrases
        
        # Individual word matching - more generous
        for word in question_words:
            if len(word) >= 2:  # Lowered from 3 to 2
                # Count occurrences
                word_count = len(re.findall(r'\b' + re.escape(word) + r'\b', content_lower))
                if word_count > 0:
                    score += word_count * 3  # Higher base score
                
                # Partial word matching for longer words
                if len(word) >= 4:
                    partial_matches = len(re.findall(re.escape(word), content_lower))
                    score += partial_matches * 1
        
        # Title matching bonus - very high priority
        title = post.get('title', '').lower()
        if title:
            for word in question_words:
                if len(word) >= 2 and word in title:
                    score += 15  # Very high bonus for title matches
        
        # Keyword-based bonuses for specific topics
        if any(word in question_lower for word in ['gpt', 'model', 'turbo', 'api', 'openai']):
            if any(term in content_lower for term in ['gpt', 'model', 'turbo', 'api', 'openai', 'ai-proxy']):
                score += 10
        
        if any(word in question_lower for word in ['docker', 'podman', 'container']):
            if any(term in content_lower for term in ['docker', 'podman', 'container']):
                score += 10
        
        if any(word in question_lower for word in ['dashboard', 'score', 'grade', 'ga4', 'bonus']):
            if any(term in content_lower for term in ['dashboard', 'score', 'grade', 'ga4', 'bonus']):
                score += 10
        
        # Special patterns
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', content_text):  # Dates
            if any(word in question_lower for word in ['date', 'when', 'schedule', 'exam', 'deadline']):
                score += 8
                
        if 'http' in content_text and 'link' in question_lower:
            score += 5
        
        # Much lower threshold for inclusion - include even weak matches
        if score > 0:
            results.append({
                'content': content_text,
                'score': score,
                'source': 'discourse',
                'title': post.get('title', 'Discussion'),
                'url': post.get('url', ''),
                'type': 'post',
                'raw_data': post
            })
    
    # Search course content with same generous approach
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
                    score += 15
        
        # Individual word matching - more generous
        for word in question_words:
            if len(word) >= 2:
                word_count = len(re.findall(r'\b' + re.escape(word) + r'\b', content_lower))
                if word_count > 0:
                    score += word_count * 2
                
                # Partial matching for longer words
                if len(word) >= 4:
                    partial_matches = len(re.findall(re.escape(word), content_lower))
                    score += partial_matches * 1
        
        # Title matching bonus
        title = content_item.get('title', '').lower()
        if title:
            for word in question_words:
                if len(word) >= 2 and word in title:
                    score += 12
        
        # Keyword-based bonuses
        if any(word in question_lower for word in ['gpt', 'model', 'turbo', 'api', 'openai']):
            if any(term in content_lower for term in ['gpt', 'model', 'turbo', 'api', 'openai', 'ai-proxy']):
                score += 8
        
        if any(word in question_lower for word in ['docker', 'podman', 'container']):
            if any(term in content_lower for term in ['docker', 'podman', 'container']):
                score += 8
        
        if any(word in question_lower for word in ['dashboard', 'score', 'grade', 'ga4', 'bonus']):
            if any(term in content_lower for term in ['dashboard', 'score', 'grade', 'ga4', 'bonus']):
                score += 8
        
        # Special patterns
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', content_text):  # Dates
            if any(word in question_lower for word in ['date', 'when', 'schedule', 'exam', 'deadline']):
                score += 6
                
        if score > 0:
            results.append({
                'content': content_text,
                'score': score,
                'source': 'course',
                'title': content_item.get('title', 'Course Content'),
                'type': 'content',
                'raw_data': content_item
            })
    
    # Sort by score and return top results
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def build_context(question: str, max_chars: int = 8000) -> tuple[str, List[Dict]]:
    """Build context string from search results with improved link handling"""
    search_results = search_content(question)
    
    print(f"Question: {question}")
    print(f"Found {len(search_results)} total results")
    
    if not search_results:
        print(f"No search results found for: {question}")
        return "", []
    
    # Show top results for debugging
    for i, result in enumerate(search_results[:5]):
        print(f"  Result {i+1}: Score {result['score']}, Source: {result['source']}, Title: {result.get('title', 'No title')[:50]}")
        print(f"    Content preview: {result['content'][:100]}...")
    
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
        
        # Always try to collect links - improved link handling
        link_url = None
        link_text = title or 'Related Content'
        
        # For discourse posts, try multiple ways to get URL
        if result['source'] == 'discourse':
            raw_data = result.get('raw_data', {})
            # Try different URL fields that might exist in discourse data
            link_url = (raw_data.get('url') or 
                       raw_data.get('link') or 
                       raw_data.get('discourse_url') or
                       raw_data.get('post_url'))
            
            # If no direct URL, try to construct one from available data
            if not link_url and raw_data.get('id'):
                # This is a fallback - you might need to adjust based on your discourse setup
                link_url = f"#discourse-post-{raw_data['id']}"
        
        # For course content, try to get any available URL
        elif result['source'] == 'course':
            raw_data = result.get('raw_data', {})
            link_url = (raw_data.get('url') or 
                       raw_data.get('link') or 
                       raw_data.get('content_url'))
        
        # Add link if we have one, or create a reference link
        if link_url:
            links.append({
                'url': link_url,
                'text': link_text
            })
        else:
            # Even without URL, provide a reference
            links.append({
                'url': '#',
                'text': f"{link_text} (Reference)"
            })
    
    # Always ensure we have at least one link if we have content
    if context_parts and not links:
        links.append({
            'url': '#course-materials',
            'text': 'Course Materials Reference'
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
                "links": [{'url': '#', 'text': 'Course Materials'}]
            }
        
        # Build context from search results
        context, links = build_context(query.question)
        
        if not context.strip():
            # Even if no context, provide a helpful response with links
            return {
                "answer": "I couldn't find specific information about your question in the course materials. Please check the course content or contact the instructor for more details.",
                "links": [
                    {'url': '#course-materials', 'text': 'Course Materials'},
                    {'url': '#discourse-discussions', 'text': 'Course Discussions'}
                ]
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
                max_tokens=200  # Limit response length
            )
            
            if response.choices and response.choices[0].message.content:
                answer = response.choices[0].message.content.strip()
                
                # Ensure we always return links
                if not links:
                    links = [{'url': '#course-materials', 'text': 'Related Course Materials'}]
                
                return {
                    "answer": answer,
                    "links": links[:5]  # Limit links
                }
            else:
                return {
                    "answer": "I couldn't generate a response. Please try rephrasing your question.",
                    "links": links if links else [{'url': '#', 'text': 'Course Materials'}]
                }
                
        except Exception as api_error:
            print(f"OpenAI API error: {str(api_error)}")
            return {
                "answer": "I'm having trouble processing your question right now. Please try again later.",
                "links": links if links else [{'url': '#', 'text': 'Course Materials'}]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in answer_question: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)