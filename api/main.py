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
        "You are helpful and informative, providing detailed answers based on course materials. "
        "Your responses must follow these guidelines:\n\n"
        "1. CONTEXT UTILIZATION:\n"
        "   - Use information from the provided context as your primary source\n"
        "   - When specific details aren't available, provide general guidance based on course principles\n"
        "   - Use your knowledge to explain concepts mentioned in the context\n\n"
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

def find_relevant_context(question: str, max_results: int = 5) -> Tuple[str, List[Dict]]:
    """Find relevant context from course content and discourse posts."""
    try:
        relevant_posts = []
        
        # Prepare question for matching
        question_lower = question.lower()
        # Break into word groups for better matching
        question_words = question_lower.split()
        question_bigrams = [f"{question_words[i]} {question_words[i+1]}" for i in range(len(question_words)-1)]
        question_trigrams = [f"{question_words[i]} {question_words[i+1]} {question_words[i+2]}" for i in range(len(question_words)-2)]
        
        # Check for exact URL matches first
        url_matches = []
        for post in discourse_posts:
            if isinstance(post, dict) and "url" in post:
                if post["url"] in question:
                    url_matches.append(post)
                    
        # Score calculation helper
        def calculate_match_score(text: str, title: str = "") -> float:
            text = text.lower()
            title = title.lower()
            score = 0.0
            
            # Word matches
            score += sum(2.0 for word in question_words if word in text)
            score += sum(3.0 for word in question_words if title and word in title)
            
            # Phrase matches (weighted higher)
            score += sum(5.0 for bigram in question_bigrams if bigram in text)
            score += sum(7.0 for bigram in question_bigrams if title and bigram in title)
            score += sum(10.0 for trigram in question_trigrams if trigram in text)
            score += sum(15.0 for trigram in question_trigrams if title and trigram in title)
            
            # Boost score if all question words appear
            if all(word in text or (title and word in title) for word in question_words):
                score *= 1.5
                
            return score
        
        # Process all discourse posts
        post_scores = []
        for idx, post in enumerate(discourse_posts):
            if not isinstance(post, dict):
                continue
                
            text = post.get("text", "")
            title = post.get("title", "")
            
            # Calculate comprehensive match score
            score = calculate_match_score(text, title)
            
            if score > 0:
                post_scores.append((score, idx, post))
        
        # Combine URL matches and scored matches
        if url_matches:
            relevant_posts.extend(url_matches)
            max_results -= len(url_matches)
            
        # Add top scoring posts
        if max_results > 0:
            sorted_posts = sorted(post_scores, key=lambda x: (-x[0], x[1]))
            relevant_posts.extend([post for _, _, post in sorted_posts[:max_results]])
        
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

import base64
import re

class Query(BaseModel):
    question: str
    image: Optional[str] = None
    
    def process_image(self) -> Optional[str]:
        """Process and validate image data"""
        if not self.image:
            return None
            
        # Handle base64 data
        try:
            # Check if it's a base64 string
            if re.match(r'^data:image/[a-zA-Z]+;base64,', self.image):
                # It's already in the correct format
                return self.image
            elif re.match(r'^[A-Za-z0-9+/]+={0,2}$', self.image):
                # It's base64 without the prefix, add it
                return f"data:image/png;base64,{self.image}"
            else:
                # Try to decode and re-encode to validate
                base64.b64decode(self.image)
                return f"data:image/png;base64,{self.image}"
        except:
            print("Warning: Invalid base64 image data received")
            return None

@app.post("/api/")
async def answer_question(query: Query):
    try:
        print(f"Received question: {query.question}")
        
        # Get relevant context and posts
        context, relevant_posts = find_relevant_context(query.question)
        
        # Process the question and get modified context and additional docs
        processed_context, additional_docs = process_question(query.question, context)
          # Process image data if present
        image_data = query.process_image() if query.image else None
        
        # Build context string with metadata
        context_string = processed_context
        if image_data:
            # For base64 images, acknowledge their presence without including the data
            context_string = "Note: An image was provided with this question.\n\n" + context_string
          # Prepare messages for the API
        messages = [
            {
                "role": "system",
                "content": get_system_prompt()
            },
            {
                "role": "user",
                "content": f"""Context from course materials and discussions:
{context_string}

Question: {query.question}

Please provide a direct answer based on the available information. Focus on being helpful while maintaining accuracy."""
            }
        ]
        
        # Add image as a separate message if present
        if image_data:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the image related to the question:"},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
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
      # Enhance context with course-specific knowledge when needed
    if not context:
        base_context = "This is regarding the Tools in Data Science course at IIT Madras, a practical diploma-level course. "
        
        if QuestionType.TOOL_CHOICE in question_types:
            context = base_context + "The course emphasizes using modern tools and best practices in data science."
        elif QuestionType.MODEL_CHOICE in question_types:
            context = base_context + "The course uses various AI models and tools for data science tasks."
        elif QuestionType.EXAM_SCHEDULE in question_types:
            context = base_context + "The course includes various assessments including gradable assignments (GA) and examinations."
        elif QuestionType.GRADING in question_types:
            context = base_context + "The course uses a comprehensive grading system including regular assessments and bonus points."
    
    # Add hints for common scenarios
    if QuestionType.TOOL_CHOICE in question_types:
        context += "\nThe course encourages using industry-standard tools and following best practices."
    elif QuestionType.MODEL_CHOICE in question_types:
        context += "\nThe course recommends using the most appropriate and efficient tools for each task."
    
    return context, additional_docs