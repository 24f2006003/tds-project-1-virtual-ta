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

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set")

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
        
        print(f"Loading course content from: {course_content_path}")
        print(f"Loading discourse posts from: {discourse_posts_path}")
        
        with open(course_content_path, "r", encoding="utf-8") as f:
            course_content = json.load(f)
        
        with open(discourse_posts_path, "r", encoding="utf-8") as f:
            discourse_posts = json.load(f)
        
        print(f"Successfully loaded {len(course_content)} course content entries")
        print(f"Successfully loaded {len(discourse_posts)} discourse posts")
        
        # Debug: Print structure of first few entries
        if course_content:
            print("Course content structure:")
            for i, item in enumerate(course_content[:2]):
                print(f"  Entry {i}: {type(item)} - Keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}")
                if isinstance(item, dict) and 'text' in item:
                    print(f"    Text length: {len(item['text'])}")
                    print(f"    Text preview: {item['text'][:100]}...")
        
        if discourse_posts:
            print("Discourse posts structure:")
            for i, item in enumerate(discourse_posts[:2]):
                print(f"  Entry {i}: {type(item)} - Keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}")
                if isinstance(item, dict):
                    if 'text' in item:
                        print(f"    Text length: {len(item['text'])}")
                        print(f"    Text preview: {item['text'][:100]}...")
                    if 'title' in item:
                        print(f"    Title: {item['title']}")
        
        return course_content, discourse_posts
            
    except FileNotFoundError as e:
        print(f"Data file not found: {str(e)}")
        return [], []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {str(e)}")
        return [], []
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return [], []

# Load data at startup
course_content, discourse_posts = load_data()

def get_system_prompt():
    """System prompt for the Virtual TA"""
    return (
        "You are a Virtual Teaching Assistant for the Tools in Data Science course at IIT Madras. "
        "You must answer questions based on the provided context from course materials and discourse discussions. "
        "Guidelines:\n\n"
        "1. Use the provided context to answer questions accurately and helpfully\n"
        "2. If the context contains relevant information, use it to provide a comprehensive answer\n"
        "3. Quote specific details, numbers, and examples from the context when relevant\n"
        "4. If the context doesn't contain enough information to answer fully, say so and provide what you can\n"
        "5. Be direct and practical in your responses\n"
        "6. Include links when they are provided in the context\n"
        "7. Structure your answer clearly and concisely\n\n"
        "Answer based on the context provided, and be helpful to the student."
    )

def extract_searchable_text(item: Any) -> str:
    """Extract all searchable text from a JSON item"""
    if isinstance(item, str):
        return item
    elif isinstance(item, dict):
        text_parts = []
        for key, value in item.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(value)
            elif isinstance(value, (list, dict)):
                text_parts.append(extract_searchable_text(value))
        return " ".join(text_parts)
    elif isinstance(item, list):
        return " ".join([extract_searchable_text(subitem) for subitem in item])
    else:
        return str(item)

def calculate_relevance_score(text: str, question: str) -> float:
    """Calculate relevance score with more liberal matching"""
    if not text or not question:
        return 0.0
    
    text_lower = text.lower()
    question_lower = question.lower()
    
    # Extract meaningful words from question (length >= 3)
    question_words = [w for w in re.findall(r'\b\w{3,}\b', question_lower)]
    if not question_words:
        return 0.0
    
    score = 0.0
    
    # Direct phrase matching (highest score)
    for i in range(len(question_words) - 1):
        phrase = f"{question_words[i]} {question_words[i+1]}"
        if phrase in text_lower:
            score += 10.0
    
    # Individual word matching
    text_words = set(re.findall(r'\b\w+\b', text_lower))
    word_matches = sum(1 for word in question_words if word in text_words)
    score += word_matches * 2.0
    
    # Technical terms (with dots, dashes, numbers)
    tech_terms = re.findall(r'\b[a-zA-Z0-9]+[.-][a-zA-Z0-9.-]+\b', text_lower)
    for term in tech_terms:
        if term in question_lower:
            score += 5.0
    
    # Numbers and versions
    numbers = re.findall(r'\b\d+(?:[./]\d+)*\b', text_lower)
    for num in numbers:
        if num in question_lower:
            score += 3.0
    
    # URLs
    urls = re.findall(r'https?://[^\s]+', text_lower)
    for url in urls:
        if url in question_lower:
            score += 15.0  # Very high score for direct URL matches
    
    # Semantic keywords
    semantic_keywords = {
        'install': ['setup', 'installation', 'installing'],
        'error': ['issue', 'problem', 'trouble', 'fail'],
        'version': ['release', 'update'],
        'configure': ['configuration', 'config', 'setting'],
        'run': ['execute', 'start', 'launch']
    }
    
    for question_word in question_words:
        if question_word in semantic_keywords:
            for synonym in semantic_keywords[question_word]:
                if synonym in text_lower:
                    score += 1.5
    
    # Coverage bonus
    if question_words:
        coverage = len(set(question_words) & text_words) / len(question_words)
        score += coverage * 5.0
    
    return score

def find_relevant_context(question: str, max_chars: int = 4000) -> tuple[str, List[Dict]]:
    """Find relevant context from JSON files with better extraction"""
    try:
        print(f"Searching for context for question: {question}")
        
        all_content = []
        relevant_links = []
        
        # Process discourse posts
        print(f"Processing {len(discourse_posts)} discourse posts...")
        for idx, post in enumerate(discourse_posts):
            if not isinstance(post, dict):
                continue
            
            # Extract all text content from the post
            full_text = extract_searchable_text(post)
            
            if not full_text.strip():
                continue
            
            score = calculate_relevance_score(full_text, question)
            
            if score > 0:
                all_content.append({
                    'text': full_text,
                    'score': score,
                    'source': 'discourse',
                    'title': post.get('title', ''),
                    'url': post.get('url', ''),
                    'index': idx
                })
                
                # Add link if available
                if post.get('url'):
                    relevant_links.append({
                        'url': post['url'],
                        'text': post.get('title', 'Discussion')
                    })
        
        # Process course content
        print(f"Processing {len(course_content)} course content entries...")
        for idx, content in enumerate(course_content):
            if not isinstance(content, dict):
                continue
            
            # Extract all text content
            full_text = extract_searchable_text(content)
            
            if not full_text.strip():
                continue
            
            score = calculate_relevance_score(full_text, question)
            
            if score > 0:
                all_content.append({
                    'text': full_text,
                    'score': score,
                    'source': 'course',
                    'title': content.get('title', ''),
                    'index': idx
                })
        
        # Sort by relevance score
        all_content.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"Found {len(all_content)} relevant content pieces")
        for i, content in enumerate(all_content[:5]):
            print(f"  #{i+1}: Score {content['score']:.1f} from {content['source']} - {content['title'][:50]}...")
        
        # Build context string within character limit
        context_parts = []
        total_chars = 0
        
        for content in all_content:
            text = content['text']
            
            # Add source information
            source_info = f"[{content['source'].upper()}]"
            if content['title']:
                source_info += f" {content['title']}"
            
            content_block = f"{source_info}\n{text}\n"
            
            if total_chars + len(content_block) > max_chars:
                # Try to fit a truncated version
                remaining_chars = max_chars - total_chars - len(source_info) - 20
                if remaining_chars > 100:
                    truncated_text = text[:remaining_chars] + "..."
                    content_block = f"{source_info}\n{truncated_text}\n"
                    context_parts.append(content_block)
                break
            
            context_parts.append(content_block)
            total_chars += len(content_block)
        
        combined_context = "\n---\n".join(context_parts)
        
        print(f"Built context with {len(combined_context)} characters from {len(context_parts)} sources")
        
        # Remove duplicate links
        unique_links = []
        seen_urls = set()
        for link in relevant_links:
            if link['url'] not in seen_urls:
                unique_links.append(link)
                seen_urls.add(link['url'])
        
        return combined_context, unique_links[:10]  # Limit links
        
    except Exception as e:
        print(f"Error in find_relevant_context: {str(e)}")
        import traceback
        traceback.print_exc()
        return "", []

class Query(BaseModel):
    question: str
    image: Optional[str] = None
    
    def process_image(self) -> Optional[str]:
        """Process and validate base64 image data"""
        if not self.image:
            return None
        
        try:
            if self.image.startswith('data:image/'):
                return self.image
            elif re.match(r'^[A-Za-z0-9+/]+={0,2}$', self.image):
                base64.b64decode(self.image)
                return f"data:image/png;base64,{self.image}"
            else:
                return None
        except Exception as e:
            print(f"Invalid image data: {str(e)}")
            return None

@app.post("/api/")
async def answer_question(query: Query):
    """Main endpoint to answer questions"""
    try:
        print(f"\n=== Processing question: {query.question} ===")
        
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Check if we have data
        if not course_content and not discourse_posts:
            return {
                "answer": "No course materials are currently available. Please check if the data files are loaded properly.",
                "links": []
            }
        
        # Find relevant context
        context, links = find_relevant_context(query.question)
        
        print(f"Context found: {len(context)} characters")
        print(f"Links found: {len(links)}")
        
        # Always try to provide an answer if we have any data
        if not context.strip():
            # If no specific context found, use a sample of available data
            sample_context = ""
            if course_content:
                sample_text = extract_searchable_text(course_content[0])
                sample_context += f"[COURSE CONTENT SAMPLE]\n{sample_text[:500]}...\n"
            if discourse_posts:
                sample_text = extract_searchable_text(discourse_posts[0])
                sample_context += f"[DISCOURSE SAMPLE]\n{sample_text[:500]}...\n"
            context = sample_context
        
        # Process image if provided
        image_data = query.process_image()
        
        # Build messages for OpenAI
        messages = [
            {
                "role": "system",
                "content": get_system_prompt()
            },
            {
                "role": "user",
                "content": f"""Here is the context from the Tools in Data Science course materials:

{context}

Student Question: {query.question}

Please provide a helpful answer based on the context provided. If the context doesn't fully address the question, provide what information you can and indicate what might be missing."""
            }
        ]
        
        # Add image if provided
        if image_data:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is an image related to the question:"},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            })
        
        # Call OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=messages,
                temperature=0.1,
                max_tokens=800
            )
            
            if response.choices and response.choices[0].message.content:
                answer = response.choices[0].message.content.strip()
                print(f"Generated answer: {len(answer)} characters")
                
                return {
                    "answer": answer,
                    "links": links
                }
            else:
                return {
                    "answer": "I was unable to generate a response. Please try rephrasing your question.",
                    "links": []
                }
                
        except Exception as api_error:
            print(f"OpenAI API error: {str(api_error)}")
            return {
                "answer": "I'm having trouble processing your question right now. Please try again later.",
                "links": links
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in answer_question: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "course_content_entries": len(course_content),
        "discourse_posts_entries": len(discourse_posts),
        "total_data_sources": len(course_content) + len(discourse_posts),
        "openai_configured": bool(OPENAI_API_KEY)
    }

@app.get("/api/debug")
async def debug_data():
    """Debug endpoint to see what data is actually loaded"""
    debug_info = {
        "course_content": {
            "count": len(course_content),
            "samples": []
        },
        "discourse_posts": {
            "count": len(discourse_posts),
            "samples": []
        }
    }
    
    # Sample course content
    for i, item in enumerate(course_content[:3]):
        if isinstance(item, dict):
            sample = {
                "index": i,
                "type": type(item).__name__,
                "keys": list(item.keys()),
                "text_preview": extract_searchable_text(item)[:200] + "..." if extract_searchable_text(item) else "No text"
            }
            debug_info["course_content"]["samples"].append(sample)
    
    # Sample discourse posts
    for i, item in enumerate(discourse_posts[:3]):
        if isinstance(item, dict):
            sample = {
                "index": i,
                "type": type(item).__name__,
                "keys": list(item.keys()),
                "title": item.get("title", "No title")[:100],
                "has_url": bool(item.get("url")),
                "text_preview": extract_searchable_text(item)[:200] + "..." if extract_searchable_text(item) else "No text"
            }
            debug_info["discourse_posts"]["samples"].append(sample)
    
    return debug_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)