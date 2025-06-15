from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
import os
import json
from openai import OpenAI
import re

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

# Topic patterns for context matching
topic_patterns = {
    'technical_tools': [
        r'\b(?:tool|software|package|library|framework)\b',
        r'\b(?:install|setup|configure|run)\b',
        r'\b(?:vs\.|versus|or|compare|better)\b'
    ],
    'course_schedule': [
        r'\b(?:date|time|deadline|when|schedule)\b',
        r'\b(?:exam|test|assignment|ga\d+)\b',
        r'\b(?:due|submission|end)\b'
    ],
    'grading': [
        r'\b(?:grade|score|mark|point|credit)\b',
        r'\b(?:bonus|extra|additional)\b',
        r'\b(?:dashboard|report|display)\b'
    ],
    'technical_requirements': [
        r'\b(?:version|compatibility|support)\b',
        r'\b(?:require|need|must|should)\b',
        r'\b(?:api|model|engine)\b'
    ],
    'course_content': [
        r'\b(?:lecture|material|content|topic)\b',
        r'\b(?:cover|discuss|explain|mean)\b',
        r'\b(?:concept|theory|practice)\b'
    ]
}

def analyze_question_topics(question: str) -> Dict[str, float]:
    """
    Analyze the question to identify relevant topics and their importance
    Returns a dictionary of topics and their relevance scores
    """
    question_lower = question.lower()
    topic_scores = {}
    
    # Score each topic based on pattern matches
    for topic, patterns in topic_patterns.items():
        score = 0.0
        for pattern in patterns:
            matches = len(re.findall(pattern, question_lower))
            if matches > 0:
                score += matches * 0.5
        if score > 0:
            topic_scores[topic] = score
            
    return topic_scores

def get_system_prompt():
    return (
        "You are a Virtual Teaching Assistant for the Tools in Data Science course at IIT Madras. "
        "You are helpful, informative, and accurate in your responses. Your goal is to assist students "
        "by providing clear, specific guidance based on course materials. Follow these guidelines:\n\n"
        "1. RESPONSE APPROACH:\n"
        "   - Be direct and specific in your answers\n"
        "   - When technical details are mentioned in context (versions, tools, numbers), use them exactly\n"
        "   - Explain your recommendations, don't just state them\n\n"
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

def score_text_relevance(text: str, title: str, question: str, question_topics: Dict[str, float]) -> float:
    """Score how relevant a text is to the question"""
    if not text:
        return 0.0
        
    text_lower = text.lower()
    title_lower = title.lower() if title else ""
    question_lower = question.lower()
    
    # Get question words (excluding very short words)
    question_words = [w for w in question_lower.split() if len(w) > 2]
    
    # Basic text matching
    text_words = set(text_lower.split())
    title_words = set(title_lower.split()) if title else set()
    
    score = 0.0
    
    # Word matching score
    word_matches = sum(1 for w in question_words if w in text_words)
    title_matches = sum(2 for w in question_words if w in title_words)  # Title matches count double
    score += word_matches + title_matches
    
    # Topic relevance score
    for topic, topic_score in question_topics.items():
        if any(re.search(pattern, text_lower) for pattern in topic_patterns[topic]):
            score += topic_score * 1.5  # Topic matches are important
            
    # Special term matching
    tech_terms = set(re.findall(r'\b[a-zA-Z]+[-\.][a-zA-Z0-9\.-]+\b', text))  # Find version numbers, technical terms
    numbers = set(re.findall(r'\b\d+(?:[\/\.]\d+)*\b', text))  # Find numeric values
    
    score += sum(3.0 for term in tech_terms if term.lower() in question_lower)  # Technical terms are very important
    score += sum(2.0 for num in numbers if num in question_lower)  # Numeric matches are important
    
    # Coverage bonus
    word_coverage = len(set(question_words) & text_words) / len(question_words)
    if word_coverage > 0.5:
        score *= (1.0 + word_coverage)  # Boost score if many question words are found
        
    return score

def find_relevant_context(question: str, max_results: int = 5) -> Tuple[str, List[Dict]]:
    """Find relevant context from course content and discourse posts."""
    try:
        # Initial analysis
        question_topics = analyze_question_topics(question)
        
        # Process all discourse posts
        scored_posts = []
        for idx, post in enumerate(discourse_posts):
            if not isinstance(post, dict):
                continue
                
            # High score for exact URL matches
            if "url" in post and post["url"] in question:
                scored_posts.append((100.0, idx, post))
                continue
                
            # Score post content
            text = post.get("text", "")
            title = post.get("title", "")
            score = score_text_relevance(text, title, question, question_topics)
            
            if score > 0:
                scored_posts.append((score, idx, post))
        
        # Sort posts by relevance score and get top matches
        sorted_posts = sorted(scored_posts, key=lambda x: (-x[0], x[1]))
        relevant_posts = [post for _, _, post in sorted_posts[:max_results]]
        
        # Get course content with score
        course_context = ""
        if course_content and len(course_content) > 0:
            text = course_content[0].get("text", "")
            score = score_text_relevance(text, "", question, question_topics)
            if score > 0:
                course_context = text[:2000]  # Keep substantial context
                
        # Combine contexts intelligently
        contexts = []
        if course_context:
            contexts.append(course_context)
            
        for post in relevant_posts:
            if isinstance(post, dict):
                post_text = post.get("text", "")
                # Include more context for highly relevant posts
                if post.get("url", "") in question:
                    contexts.append(post_text)  # Full text for URL matches
                else:
                    contexts.append(post_text[:500])  # Limited context for others
                    
        combined_context = "\n\n".join(contexts)
        return combined_context, relevant_posts
    except Exception as e:
        print(f"Error in find_relevant_context: {str(e)}")
        return "", []

import base64

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
    
    # Keep existing context if available
    if not context:
        context = ""
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

# Define topic patterns at module level for reuse
topic_patterns = {
        'technical_tools': [
            r'\b(?:tool|software|package|library|framework)\b',
            r'\b(?:install|setup|configure|run)\b',
            r'\b(?:vs\.|versus|or|compare|better)\b'
        ],
        'course_schedule': [
            r'\b(?:date|time|deadline|when|schedule)\b',
            r'\b(?:exam|test|assignment|ga\d+)\b',
            r'\b(?:due|submission|end)\b'
        ],
        'grading': [
            r'\b(?:grade|score|mark|point|credit)\b',
            r'\b(?:bonus|extra|additional)\b',
            r'\b(?:dashboard|report|display)\b'
        ],
        'technical_requirements': [
            r'\b(?:version|compatibility|support)\b',
            r'\b(?:require|need|must|should)\b',
            r'\b(?:api|model|engine)\b'
        ],
        'course_content': [
            r'\b(?:lecture|material|content|topic)\b',
            r'\b(?:cover|discuss|explain|mean)\b',
            r'\b(?:concept|theory|practice)\b'
        ]    }