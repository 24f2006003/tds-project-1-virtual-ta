from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import re
import unicodedata
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
        "Answer questions using ONLY the provided context from course materials and discussion posts. "
        "Be helpful, concise, and accurate.\n\n"
        "Guidelines:\n"
        "1. Use information directly from the provided context\n"
        "2. If the exact information isn't available, say 'This specific information is not available in the current course materials'\n"
        "3. Provide practical, actionable answers when possible\n"
        "4. Reference relevant sections or topics when helpful\n"
        "5. Keep responses focused and under 150 words unless more detail is specifically needed\n"
    )

def normalize_text(text: str) -> str:
    """Normalize text for better matching"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    return text

def extract_searchable_text(item: Any, max_depth: int = 5) -> str:
    """Recursively extract all searchable text content from nested JSON structures"""
    if max_depth <= 0:
        return ""
    
    if isinstance(item, str):
        return item.strip()
    elif isinstance(item, dict):
        texts = []
        # Process all values, but prioritize certain keys
        priority_keys = ['title', 'content', 'text', 'body', 'description', 'summary']
        other_keys = [k for k in item.keys() if k.lower() not in [pk.lower() for pk in priority_keys] 
                     and k.lower() not in ['id', 'index', 'timestamp', 'url', 'slug']]
        
        # Process priority keys first
        for key in priority_keys:
            if key in item:
                text = extract_searchable_text(item[key], max_depth - 1)
                if text:
                    texts.append(text)
        
        # Then process other keys
        for key in other_keys:
            text = extract_searchable_text(item[key], max_depth - 1)
            if text:
                texts.append(text)
        
        return " ".join(texts)
    elif isinstance(item, list):
        texts = []
        for subitem in item:
            text = extract_searchable_text(subitem, max_depth - 1)
            if text:
                texts.append(text)
        return " ".join(texts)
    else:
        return str(item) if item is not None else ""

def calculate_relevance_score(question_words: List[str], content: str) -> float:
    """Calculate relevance score based on word matches and their frequency"""
    if not content or not question_words:
        return 0.0
    
    content_normalized = normalize_text(content)
    score = 0.0
    
    for word in question_words:
        if len(word) < 3:  # Skip very short words
            continue
        
        word_normalized = normalize_text(word)
        
        # Count exact matches
        exact_matches = content_normalized.count(word_normalized)
        score += exact_matches * 2.0
        
        # Check for partial matches
        if word_normalized in content_normalized:
            score += 1.0
        
        # Boost score for matches in what appears to be titles or headers
        if any(phrase in content_normalized for phrase in [f"# {word_normalized}", f"## {word_normalized}", f"### {word_normalized}"]):
            score += 3.0
    
    # Normalize by content length to avoid bias towards longer content
    content_length_factor = min(1.0, len(content) / 1000)
    return score * content_length_factor

def enhanced_search(question: str) -> List[Dict]:
    """Enhanced search that finds relevant content more effectively"""
    question_normalized = normalize_text(question)
    question_words = [w for w in question_normalized.split() if len(w) > 2]
    
    if not question_words:
        return []
    
    results = []
    
    print(f"Searching for: '{question}'")
    print(f"Question words: {question_words}")
    print(f"Available discourse posts: {len(discourse_posts)}")
    print(f"Available course content: {len(course_content)}")
    
    # Search discourse posts
    for idx, post in enumerate(discourse_posts):
        try:
            searchable_text = extract_searchable_text(post)
            if not searchable_text:
                continue
            
            relevance_score = calculate_relevance_score(question_words, searchable_text)
            
            if relevance_score > 0:
                results.append({
                    'content': searchable_text[:2000],  # Limit content length
                    'source': 'discourse',
                    'title': post.get('title', 'Discussion Post'),
                    'url': post.get('url', '#'),
                    'score': relevance_score,
                    'raw_data': post
                })
                print(f"Found discourse match (score: {relevance_score:.2f}): {post.get('title', 'No title')[:50]}")
        
        except Exception as e:
            print(f"Error processing discourse post {idx}: {e}")
            continue
    
    # Search course content
    for idx, content_item in enumerate(course_content):
        try:
            searchable_text = extract_searchable_text(content_item)
            if not searchable_text:
                continue
            
            relevance_score = calculate_relevance_score(question_words, searchable_text)
            
            if relevance_score > 0:
                results.append({
                    'content': searchable_text[:2000],  # Limit content length
                    'source': 'course',
                    'title': content_item.get('title', 'Course Content'),
                    'url': content_item.get('url', '#'),
                    'score': relevance_score,
                    'raw_data': content_item
                })
                print(f"Found course match (score: {relevance_score:.2f}): {content_item.get('title', 'No title')[:50]}")
        
        except Exception as e:
            print(f"Error processing course content {idx}: {e}")
            continue
    
    # Sort by relevance score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Total matches found: {len(results)}")
    if results:
        print(f"Top match score: {results[0]['score']:.2f}")
    
    return results

def build_context(question: str) -> tuple[str, List[Dict]]:
    """Build context from search results"""
    search_results = enhanced_search(question)
    
    if not search_results:
        print("No search results found")
        # Try a fallback search with more lenient criteria
        return try_fallback_search(question)
    
    context_parts = []
    links = []
    
    # Take top 5 results to avoid token limits
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
    
    context = "\n" + "="*50 + "\n".join(context_parts)
    print(f"Built context with {len(context)} characters from {len(context_parts)} sources")
    
    return context, links

def try_fallback_search(question: str) -> tuple[str, List[Dict]]:
    """Fallback search that's more lenient"""
    print("Trying fallback search...")
    
    all_content = []
    links = []
    
    # If no specific matches, try to return some general content
    if course_content:
        for i, item in enumerate(course_content[:3]):  # Take first 3 items
            content = extract_searchable_text(item)
            if content:
                all_content.append(f"[COURSE] {item.get('title', 'Course Material')}\n{content[:1000]}")
                links.append({
                    'url': item.get('url', '#'),
                    'text': item.get('title', 'Course Material')
                })
    
    if discourse_posts:
        for i, item in enumerate(discourse_posts[:3]):  # Take first 3 items
            content = extract_searchable_text(item)
            if content:
                all_content.append(f"[DISCOURSE] {item.get('title', 'Discussion Post')}\n{content[:1000]}")
                links.append({
                    'url': item.get('url', '#'),
                    'text': item.get('title', 'Discussion Post')
                })
    
    context = "\n" + "="*50 + "\n".join(all_content) if all_content else ""
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
                "answer": "No course materials are currently available. Please check if the data files are properly loaded.",
                "links": [{'url': '#', 'text': 'Course Materials'}]
            }
        
        # Build context from search results
        context, links = build_context(query.question)
        
        if not context.strip():
            return {
                "answer": "This specific information is not available in the current course materials. Please try rephrasing your question or contact the course instructor for more details.",
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
                "content": f"""Context from course materials:
{context}

Student Question: {query.question}

Please provide a helpful answer based on the context above."""
            }
        ]
        
        # Call OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=300
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
    sample_course = []
    sample_discourse = []
    
    if course_content:
        for item in course_content[:2]:
            sample_course.append({
                'keys': list(item.keys()) if isinstance(item, dict) else 'not_dict',
                'title': item.get('title', 'No title') if isinstance(item, dict) else 'No title',
                'content_preview': extract_searchable_text(item)[:200]
            })
    
    if discourse_posts:
        for item in discourse_posts[:2]:
            sample_discourse.append({
                'keys': list(item.keys()) if isinstance(item, dict) else 'not_dict',
                'title': item.get('title', 'No title') if isinstance(item, dict) else 'No title',
                'content_preview': extract_searchable_text(item)[:200]
            })
    
    return {
        "course_content_count": len(course_content),
        "discourse_posts_count": len(discourse_posts),
        "sample_course_content": sample_course,
        "sample_discourse_posts": sample_discourse
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)