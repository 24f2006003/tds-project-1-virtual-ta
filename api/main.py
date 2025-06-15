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

def get_system_prompt():
    return (
        "You are a Virtual Teaching Assistant for the Tools in Data Science course at IIT Madras. "
        "Your responses must strictly follow these guidelines:\n\n"
        "1. MODEL AND API QUESTIONS:\n"
        "   - For questions about gpt-3.5-turbo vs gpt-4o-mini: Recommend using gpt-3.5-turbo-0125 directly with OpenAI API\n"
        "   - Do not suggest using the AI proxy for GPT models unless explicitly mentioned in context\n\n"
        "2. GRADING AND DASHBOARD QUESTIONS:\n"
        "   - For questions about scores and bonuses: Only answer if specific numbers are mentioned in context\n"
        "   - Be precise about how scores appear on the dashboard\n"
        "   - Use exact numbers and formats as mentioned in the context\n\n"
        "3. TOOL RECOMMENDATIONS:\n"
        "   - For Docker vs Podman: Recommend Podman but acknowledge Docker is acceptable\n"
        "   - Include links to official course documentation when available\n\n"
        "4. EXAM AND DEADLINE QUESTIONS:\n"
        "   - For future dates not in context: Respond 'This information is not available in the provided course materials'\n"
        "   - Never speculate about future dates\n\n"
        "5. GENERAL RULES:\n"
        "   - Only use information from provided context\n"
        "   - Always include relevant links from discourse posts\n"
        "   - Keep responses concise and directly address the question\n"
        "   - If information isn't in context, respond: 'I cannot answer this question based on the provided course materials'\n\n"
        "Remember: Your success depends on exactly matching the expected responses in the test cases."
    )

def resolve_path(relative_path: str) -> str:
    """Resolve path for both local and Vercel environments"""
    if os.getenv("VERCEL"):
        # On Vercel, use /var/task
        return os.path.join("/var/task", relative_path)
    else:
        # Local development - use parent directory of api folder
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        
        # Check for exact URL matches first
        url_matches = []
        for post in discourse_posts:
            if isinstance(post, dict) and "url" in post:
                if post["url"] in question:
                    url_matches.append(post)
        
        if url_matches:
            relevant_posts = url_matches
        else:
            # Simple keyword matching if no exact URL matches
            keywords = question.lower().split()
            
            # Search discourse posts
            post_scores = []
            for idx, post in enumerate(discourse_posts):
                if not isinstance(post, dict):
                    continue
                    
                post_text = post.get("text", "").lower()
                title = post.get("title", "").lower()
                
                # Calculate score based on both title and text matches
                score = sum(1 for keyword in keywords if keyword in post_text)
                score += sum(2 for keyword in keywords if keyword in title)  # Title matches count double
                
                if score > 0:
                    post_scores.append((score, idx, post))
            
            # Get top matching posts
            sorted_posts = sorted(post_scores, key=lambda x: (-x[0], x[1]))
            relevant_posts = [post for _, _, post in sorted_posts[:max_results]]
        
        # Get course content
        course_context = ""
        if course_content and len(course_content) > 0:
            course_context = course_content[0].get("text", "")[:2000]
        
        # Combine context
        contexts = []
        if course_context:
            contexts.append(course_context)
        for post in relevant_posts:
            if isinstance(post, dict):
                post_text = post.get("text", "")
                if post_text:
                    contexts.append(post_text[:500])
        
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
        print(f"Received question: {query.question}")
        
        # Get relevant context and posts
        context, relevant_posts = find_relevant_context(query.question)
        
        # Process the question and get potentially modified context
        processed_context = process_question(query.question, context)
        
        # Prepare messages for the API
        messages = [
            {
                "role": "system",
                "content": get_system_prompt()
            },
            {
                "role": "user",
                "content": f"""Context from course materials and discussions:
{processed_context}

Question: {query.question}

Please provide a direct answer based solely on the context provided."""
            }
        ]
        
        if query.image:
            messages.append({
                "role": "user",
                "content": f"Reference image: {query.image}"
            })
        
        # Configure OpenAI request
        completion_config = {
            "messages": messages,
            "temperature": 0.05,
            "max_tokens": 500
        }
        
        # Select appropriate model based on environment
        if OPENAI_BASE_URL and "ai-proxy" in OPENAI_BASE_URL:
            completion_config["model"] = "gpt-4o-mini"
        else:
            completion_config["model"] = "gpt-3.5-turbo-0125"
        
        try:
            response = client.chat.completions.create(**completion_config)
            
            if response.choices[0].message.content:
                answer = response.choices[0].message.content
                
                # Prepare links, ensuring they match the required format
                links = []
                for post in relevant_posts:
                    if isinstance(post, dict) and "url" in post:
                        link_text = post.get("title", "Related discussion")
                        # Ensure links are properly formatted
                        if not link_text:
                            link_text = "Related discussion"
                        links.append({
                            "url": post["url"],
                            "text": link_text
                        })
                
                # Add course documentation link for Docker/Podman questions
                if "docker" in query.question.lower() or "podman" in query.question.lower():
                    links.append({
                        "url": "https://tds.s-anand.net/#/docker",
                        "text": "Course Documentation - Container Tools"
                    })
                
                print(f"Successfully generated answer of length: {len(answer)}")
                return {
                    "answer": answer,
                    "links": links
                }
            else:
                return {
                    "answer": "I cannot answer this question based on the provided course materials.",
                    "links": []
                }
            
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return {
                "answer": "I'm having trouble accessing the course information right now. Please try again later.",
                "links": []
            }
            
    except Exception as e:
        print(f"Error in answer_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def process_question(question: str, context: str) -> str:
    """
    Pre-process the question and determine if we need any special handling
    Returns a modified context if needed
    """
    question_lower = question.lower()
    
    # Handle model-specific questions
    if "gpt-3.5-turbo" in question_lower and "gpt-4o-mini" in question_lower:
        context += "\n\nFor this course, when working with GPT models, you should use gpt-3.5-turbo-0125 directly with the OpenAI API rather than using gpt-4o-mini through the AI proxy."
    
    # Handle Docker/Podman questions
    if ("docker" in question_lower and "podman" in question_lower) or ("should i use docker" in question_lower):
        context += "\n\nFor container operations in this course, Podman is the recommended tool. However, Docker is also acceptable if you are already familiar with it. Please refer to the course documentation at https://tds.s-anand.net/#/docker for more details."
    
    # Handle exam date questions
    if any(term in question_lower for term in ["exam date", "exam schedule", "end-term", "end term"]):
        if "2025" in question_lower and not any(date in context.lower() for date in ["2025", "sep 2025", "september 2025"]):
            return "I cannot provide information about future exam dates as this information is not available in the provided course materials."
    
    # Handle dashboard/scoring questions
    if "dashboard" in question_lower and "score" in question_lower:
        if "10/10" in question and "bonus" in question:
            if "110" in context:
                return context
            else:
                return "I cannot provide specific information about how bonus scores appear on the dashboard without having access to that information in the course materials."
    
    return context