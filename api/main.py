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
        "1. CONTEXT ADHERENCE:\n"
        "   - Only provide information explicitly present in the given context\n"
        "   - If information is not in context, respond: 'I cannot answer this question based on the provided course materials'\n"
        "   - Never make assumptions or use external knowledge\n\n"
        "2. TECHNICAL QUESTIONS:\n"
        "   - For tool choices: Use recommendations from course documentation\n"
        "   - For model/API choices: Refer to official course guidelines\n"
        "   - Only mention compatibility or requirements stated in context\n\n"
        "3. COURSE INFORMATION:\n"
        "   - For dates and schedules: Only cite information present in context\n"
        "   - For grading/scoring: Use exact numbers and formats from context\n"
        "   - Never speculate about future dates or unnamed tools\n\n"
        "4. RESPONSE FORMAT:\n"
        "   - Keep responses concise and directly address the question\n"
        "   - Include relevant links when available\n"
        "   - Quote specific details from context when applicable\n\n"
        "5. UNCERTAINTY HANDLING:\n"
        "   - If context is unclear: State that information is not available\n"
        "   - Don't make assumptions about course policies or requirements\n"
        "   - Better to acknowledge missing information than speculate\n\n"
        "Remember: Base all responses strictly on the provided context."
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
        
        # Process the question and get modified context and additional docs
        processed_context, additional_docs = process_question(query.question, context)
        
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

Please provide a direct answer based solely on the context provided. If the information is not in the context, state that clearly."""
            }
        ]
        
        if query.image:
            messages.append({
                "role": "user",
                "content": f"Reference image: {query.image}"
            })
        
        # Configure OpenAI request with dynamic model selection
        completion_config = {
            "messages": messages,            "temperature": 0.05,  # Low temperature for consistent responses
            "max_tokens": 500,    # Reasonable length limit
            "model": "gpt-3.5-turbo-0125"  # Using gpt-3.5-turbo-0125 as per course requirements
        }
        
        try:
            response = client.chat.completions.create(**completion_config)
            
            if response.choices[0].message.content:
                answer = response.choices[0].message.content
                
                # Prepare links from relevant posts
                links = []
                for post in relevant_posts:
                    if isinstance(post, dict) and "url" in post:
                        links.append({
                            "url": post["url"],
                            "text": post.get("title", "Related discussion")
                        })
                
                # Add any additional documentation links
                links.extend(additional_docs)
                
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

from enum import Enum
from typing import List, Dict, Optional

class QuestionType(Enum):
    TOOL_CHOICE = "tool_choice"
    MODEL_CHOICE = "model_choice"
    EXAM_SCHEDULE = "exam_schedule"
    GRADING = "grading"
    GENERAL = "general"

class QuestionAnalyzer:
    def __init__(self):
        self.patterns = {
            QuestionType.TOOL_CHOICE: {
                "keywords": [
                    ["docker", "podman"],
                    ["container", "docker"],
                    ["container", "podman"]
                ],
                "docs_url": "https://tds.s-anand.net/#/docker"
            },
            QuestionType.MODEL_CHOICE: {
                "keywords": [
                    ["gpt", "model"],
                    ["openai", "model"],
                    ["gpt-3.5", "gpt-4"],
                    ["ai-proxy", "openai"]
                ]
            },
            QuestionType.EXAM_SCHEDULE: {
                "keywords": [
                    ["exam", "date"],
                    ["exam", "schedule"],
                    ["end-term"],
                    ["end", "term"],
                    ["final", "exam"]
                ]
            },
            QuestionType.GRADING: {
                "keywords": [
                    ["score", "dashboard"],
                    ["grade", "dashboard"],
                    ["bonus", "score"],
                    ["marks", "display"]
                ]
            }
        }
    
    def identify_question_type(self, question: str) -> List[QuestionType]:
        """Identify the types of a question based on keyword patterns"""
        question_lower = question.lower()
        question_types = []
        
        for q_type, config in self.patterns.items():
            for keyword_group in config["keywords"]:
                if all(keyword in question_lower for keyword in keyword_group):
                    question_types.append(q_type)
                    break
        
        return question_types if question_types else [QuestionType.GENERAL]

    def get_relevant_docs(self, question_types: List[QuestionType]) -> List[Dict[str, str]]:
        """Get relevant documentation links for the question types"""
        docs = []
        for q_type in question_types:
            if q_type in self.patterns and "docs_url" in self.patterns[q_type]:
                docs.append({
                    "url": self.patterns[q_type]["docs_url"],
                    "text": f"Course Documentation - {q_type.value.replace('_', ' ').title()}"
                })
        return docs

def process_question(question: str, context: str) -> tuple[str, List[Dict[str, str]]]:
    """
    Process the question and determine if we need any special handling
    Returns modified context and any additional documentation links
    """
    analyzer = QuestionAnalyzer()
    question_types = analyzer.identify_question_type(question)
    additional_docs = analyzer.get_relevant_docs(question_types)
    
    # If no context is found for specific question types, provide a clear "cannot answer" response
    if not context and any(qt in [QuestionType.EXAM_SCHEDULE, QuestionType.GRADING] for qt in question_types):
        return "I cannot provide this information as it is not available in the provided course materials.", additional_docs
    
    # For future dates not in context
    if QuestionType.EXAM_SCHEDULE in question_types and "2025" in question.lower():
        if not any(date in context.lower() for date in ["2025", "sep 2025", "september 2025"]):
            return "I cannot provide information about future exam dates as this information is not available in the provided course materials.", []
    
    return context, additional_docs