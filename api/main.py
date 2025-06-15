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
    """System prompt for the Virtual TA"""
    return (
        "You are a Virtual Teaching Assistant for the Tools in Data Science course at IIT Madras. "
        "You are helpful, informative, and accurate. Follow these guidelines:\n\n"
        "1. Be direct and specific in your answers\n"
        "2. For technical tools, recommend Podman over Docker for containerization\n"
        "3. For AI models, specify GPT-3.5-turbo-0125 as the course standard\n"
        "4. Quote exact numbers and details from the provided context\n"
        "5. Include relevant links when available\n"
        "6. If information isn't in the context, say so clearly\n"
        "7. Keep responses concise and helpful\n\n"
        "Base all responses strictly on the provided context from course materials."
    )

def calculate_relevance_score(text: str, question: str) -> float:
    """Calculate how relevant a text is to the question"""
    if not text or not question:
        return 0.0
    
    text_lower = text.lower()
    question_lower = question.lower()
    
    # Get significant words from question (length > 2)
    question_words = [w for w in question_lower.split() if len(w) > 2]
    text_words = set(text_lower.split())
    
    # Basic word matching
    word_matches = sum(1 for word in question_words if word in text_words)
    
    # Technical terms and numbers get higher scores
    tech_terms = re.findall(r'\b[a-zA-Z]+[-\.][a-zA-Z0-9\.-]+\b', text_lower)
    numbers = re.findall(r'\b\d+(?:[\/\.]\d+)*\b', text_lower)
    
    tech_score = sum(2.0 for term in tech_terms if term in question_lower)
    number_score = sum(1.5 for num in numbers if num in question_lower)
    
    # Coverage bonus
    if question_words:
        coverage = len(set(question_words) & text_words) / len(question_words)
        coverage_bonus = coverage * 2.0
    else:
        coverage_bonus = 0.0
    
    return word_matches + tech_score + number_score + coverage_bonus

def find_relevant_context(question: str, max_results: int = 5) -> tuple[str, List[Dict]]:
    """Find relevant context from course content and discourse posts"""
    try:
        relevant_contexts = []
        relevant_links = []
        
        # Check for direct URL matches in question
        url_match = re.search(r'https?://[^\s]+', question)
        if url_match:
            target_url = url_match.group()
            for post in discourse_posts:
                if isinstance(post, dict) and post.get("url") == target_url:
                    relevant_contexts.append(post.get("text", "")[:1000])
                    relevant_links.append({
                        "url": post["url"],
                        "text": post.get("title", "Related discussion")
                    })
                    break
        
        # Score and rank discourse posts
        scored_posts = []
        for post in discourse_posts:
            if not isinstance(post, dict):
                continue
            
            text = post.get("text", "")
            title = post.get("title", "")
            
            # Combine title and text for scoring, giving title more weight
            combined_text = f"{title} {title} {text}"  # Title appears twice for higher weight
            score = calculate_relevance_score(combined_text, question)
            
            if score > 0:
                scored_posts.append((score, post))
        
        # Get top relevant posts
        scored_posts.sort(key=lambda x: x[0], reverse=True)
        for score, post in scored_posts[:max_results]:
            if post.get("text"):
                relevant_contexts.append(post["text"][:800])  # Limit length
            if post.get("url"):
                relevant_links.append({
                    "url": post["url"],
                    "text": post.get("title", "Related discussion")
                })
        
        # Add course content if relevant
        if course_content:
            for content in course_content:
                if isinstance(content, dict):
                    text = content.get("text", "")
                    score = calculate_relevance_score(text, question)
                    if score > 2.0:  # Higher threshold for course content
                        relevant_contexts.append(text[:1000])
                        break
        
        # Combine contexts
        combined_context = "\n\n".join(relevant_contexts)
        
        # Remove duplicate links
        unique_links = []
        seen_urls = set()
        for link in relevant_links:
            if link["url"] not in seen_urls:
                unique_links.append(link)
                seen_urls.add(link["url"])
        
        return combined_context, unique_links[:max_results]
        
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
    """Main endpoint to answer questions"""
    try:
        print(f"Received question: {query.question}")
        
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Find relevant context and links
        context, links = find_relevant_context(query.question)
        
        # Process image if provided
        image_data = query.process_image()
        
        # Prepare context for the AI
        if not context:
            context = "This question is about the Tools in Data Science course at IIT Madras."
        
        # Add specific guidance for common topics
        question_lower = query.question.lower()
        if any(word in question_lower for word in ['docker', 'podman', 'container']):
            context += "\n\nNote: For containerization, the course recommends Podman over Docker."
        
        if any(word in question_lower for word in ['gpt', 'model', 'openai']):
            context += "\n\nNote: The course uses GPT-3.5-turbo-0125 as the standard language model."
        
        # Build messages for OpenAI API
        messages = [
            {
                "role": "system",
                "content": get_system_prompt()
            },
            {
                "role": "user",
                "content": f"""Context from course materials:
{context}

Question: {query.question}

Please provide a helpful answer based on the available information."""
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
                    "answer": "I couldn't generate a response. Please try rephrasing your question.",
                    "links": []
                }
                
        except Exception as api_error:
            print(f"OpenAI API error: {str(api_error)}")
            return {
                "answer": "I'm having trouble accessing the AI service. Please try again later.",
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
        "course_content_loaded": len(course_content) > 0,
        "discourse_posts_loaded": len(discourse_posts) > 0,
        "openai_configured": bool(OPENAI_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)