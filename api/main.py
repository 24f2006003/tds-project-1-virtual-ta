from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
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
    """System prompt for the Virtual TA - only uses data from JSON files"""
    return (
        "You are a Virtual Teaching Assistant for the Tools in Data Science course at IIT Madras. "
        "You must ONLY use information provided in the context from the course materials and discourse posts. "
        "Follow these strict guidelines:\n\n"
        "1. ONLY answer based on the provided context - never add information not in the context\n"
        "2. If the answer is not in the provided context, clearly state 'This information is not available in the course materials provided'\n"
        "3. Quote exact details, numbers, and specifications from the context\n"
        "4. Do not make assumptions or provide general knowledge - stick strictly to the provided materials\n"
        "5. If multiple options are mentioned in the context, present them as they appear\n"
        "6. Include relevant links when they are provided in the context\n"
        "7. Keep responses concise and directly address the question\n\n"
        "Remember: You can ONLY use information that appears in the provided context."
    )

def calculate_relevance_score(text: str, question: str) -> float:
    """Calculate how relevant a text is to the question"""
    if not text or not question:
        return 0.0
    
    text_lower = text.lower()
    question_lower = question.lower()
    
    # Get significant words from question (length > 2)
    question_words = [w for w in question_lower.split() if len(w) > 2]
    if not question_words:
        return 0.0
    
    text_words = set(text_lower.split())
    
    # Basic word matching
    word_matches = sum(1 for word in question_words if word in text_words)
    
    # Technical terms and numbers get higher scores
    tech_terms = re.findall(r'\b[a-zA-Z]+[-\.][a-zA-Z0-9\.-]+\b', text_lower)
    numbers = re.findall(r'\b\d+(?:[\/\.]\d+)*\b', text_lower)
    
    tech_score = sum(2.0 for term in tech_terms if term in question_lower)
    number_score = sum(1.5 for num in numbers if num in question_lower)
    
    # Coverage bonus
    matching_words = set(question_words) & text_words
    coverage = len(matching_words) / len(question_words)
    coverage_bonus = coverage * 3.0
    
    return word_matches + tech_score + number_score + coverage_bonus

def find_relevant_context(question: str, max_results: int = 5) -> tuple[str, List[Dict]]:
    """Find relevant context ONLY from the loaded JSON files"""
    try:
        relevant_contexts = []
        relevant_links = []
        
        # Check for direct URL matches in discourse posts
        url_match = re.search(r'https?://[^\s]+', question)
        if url_match:
            target_url = url_match.group()
            for post in discourse_posts:
                if isinstance(post, dict) and post.get("url") == target_url:
                    text = post.get("text", "")
                    if text:
                        relevant_contexts.append(text)
                    relevant_links.append({
                        "url": post["url"],
                        "text": post.get("title", "Discussion")
                    })
                    break
        
        # Score and rank discourse posts by relevance
        scored_posts = []
        for post in discourse_posts:
            if not isinstance(post, dict):
                continue
            
            text = post.get("text", "")
            title = post.get("title", "")
            
            if not text and not title:  # Skip empty posts
                continue
            
            # Combine title and text for scoring, giving title more weight
            combined_text = f"{title} {title} {text}"  # Title appears twice for higher weight
            score = calculate_relevance_score(combined_text, question)
            
            if score > 0:
                scored_posts.append((score, post))
        
        # Get top relevant posts
        scored_posts.sort(key=lambda x: x[0], reverse=True)
        for score, post in scored_posts[:max_results]:
            text = post.get("text", "")
            url = post.get("url", "")
            title = post.get("title", "")
            
            if text and text not in relevant_contexts:  # Avoid duplicates
                relevant_contexts.append(text)
            
            if url:
                link_entry = {
                    "url": url,
                    "text": title if title else "Related discussion"
                }
                if link_entry not in relevant_links:
                    relevant_links.append(link_entry)
        
        # Score and add course content
        scored_content = []
        for content in course_content:
            if isinstance(content, dict):
                text = content.get("text", "")
                if text:
                    score = calculate_relevance_score(text, question)
                    if score > 0:
                        scored_content.append((score, text))
        
        # Add top scoring course content
        scored_content.sort(key=lambda x: x[0], reverse=True)
        for score, text in scored_content[:2]:  # Limit course content
            if text not in relevant_contexts:
                relevant_contexts.append(text)
        
        # Combine all relevant contexts
        combined_context = "\n\n---\n\n".join(relevant_contexts)
        
        return combined_context, relevant_links
        
    except Exception as e:
        print(f"Error in find_relevant_context: {str(e)}")
        return "", []

class Query(BaseModel):
    question: str
    image: Optional[str] = None
    
    def process_image(self) -> Optional[str]:
        """Process and validate base64 image data"""
        if not self.image:
            return None
        
        try:
            # Handle different base64 formats
            if self.image.startswith('data:image/'):
                return self.image
            elif re.match(r'^[A-Za-z0-9+/]+={0,2}$', self.image):
                # Validate base64 and add data URI prefix
                base64.b64decode(self.image)
                return f"data:image/png;base64,{self.image}"
            else:
                return None
        except Exception as e:
            print(f"Invalid image data: {str(e)}")
            return None

@app.post("/api/")
async def answer_question(query: Query):
    """Main endpoint to answer questions using ONLY data from JSON files"""
    try:
        print(f"Received question: {query.question}")
        
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Check if we have any data loaded
        if not course_content and not discourse_posts:
            return {
                "answer": "No course materials are currently available. Please ensure the data files are properly loaded.",
                "links": []
            }
        
        # Find relevant context ONLY from JSON files
        context, links = find_relevant_context(query.question)
        
        # If no relevant context found, be explicit about it
        if not context.strip():
            return {
                "answer": "I couldn't find relevant information for your question in the available course materials and discussions. Please try rephrasing your question or check if the topic is covered in the course.",
                "links": []
            }
        
        # Process image if provided
        image_data = query.process_image()
        
        # Build messages for OpenAI API - context comes ONLY from JSON files
        messages = [
            {
                "role": "system",
                "content": get_system_prompt()
            },
            {
                "role": "user",
                "content": f"""Context from course materials and discussions:

{context}

Question: {query.question}

Please answer based ONLY on the information provided in the context above. Do not add any external knowledge."""
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
                temperature=0.05,  # Very low temperature for consistency
                max_tokens=600
            )
            
            if response.choices and response.choices[0].message.content:
                answer = response.choices[0].message.content.strip()
                
                return {
                    "answer": answer,
                    "links": links
                }
            else:
                return {
                    "answer": "I couldn't generate a response from the available course materials.",
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

@app.get("/api/data-summary")
async def data_summary():
    """Endpoint to check what data is loaded"""
    course_sample = []
    discourse_sample = []
    
    # Get sample of course content
    for i, content in enumerate(course_content[:3]):
        if isinstance(content, dict):
            sample = {
                "index": i,
                "keys": list(content.keys()),
                "text_length": len(content.get("text", ""))
            }
            course_sample.append(sample)
    
    # Get sample of discourse posts
    for i, post in enumerate(discourse_posts[:3]):
        if isinstance(post, dict):
            sample = {
                "index": i,
                "keys": list(post.keys()),
                "title": post.get("title", "")[:50] + "..." if post.get("title", "") else "",
                "text_length": len(post.get("text", "")),
                "has_url": bool(post.get("url"))
            }
            discourse_sample.append(sample)
    
    return {
        "course_content": {
            "total_entries": len(course_content),
            "sample": course_sample
        },
        "discourse_posts": {
            "total_entries": len(discourse_posts),
            "sample": discourse_sample
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)