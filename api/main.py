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
        # Log incoming request
        print(f"Received question: {query.question}")
        
        # Get relevant context and posts
        context, relevant_posts = find_relevant_context(query.question)
        
        # Log context found
        print(f"Found {len(relevant_posts)} relevant posts")
        print(f"Context length: {len(context)} characters")
        # Prepare conversation with context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Virtual Teaching Assistant for the Tools in Data Science course at IIT Madras. "
                    "Your role is EXTREMELY RESTRICTED - you can ONLY provide information that is explicitly "
                    "present in the provided course context.\n\n"
                    "ABSOLUTE RULES - VIOLATION WILL RESULT IN FAILURE:\n\n"
                    "1. CONTEXT-ONLY RESPONSES:\n"
                    "   - You can ONLY use information explicitly stated in the provided context\n"
                    "   - NEVER use your general knowledge, training data, or external information\n"
                    "   - If the context doesn't contain the answer, you MUST say: 'I cannot answer this question based on the provided course materials.'\n"
                    "   - Do NOT paraphrase, infer, or expand beyond what's literally written in the context\n\n"
                    "2. TECHNICAL RECOMMENDATIONS:\n"
                    "   - NEVER recommend tools, software, or approaches unless explicitly mentioned in the context\n"
                    "   - For questions like 'Should I use Docker or Podman?' - only answer if the context specifically discusses this choice\n"
                    "   - If context doesn't mention a tool, respond: 'This topic is not covered in the provided course materials.'\n"
                    "   - Do NOT provide general advice about programming, data science, or software choices\n\n"
                    "3. DATES, SCHEDULES, AND DEADLINES:\n"
                    "   - Only mention dates/deadlines if explicitly stated in the provided context\n"
                    "   - NEVER provide general academic calendar information\n"
                    "   - If asked about exam dates and context doesn't contain them, say: 'Exam scheduling information is not available in the provided materials.'\n"
                    "   - Do NOT say dates 'will be announced later' unless the context specifically states this\n\n"
                    "4. GRADING AND ASSESSMENT:\n"
                    "   - Only describe grading behavior if explicitly detailed in the context\n"
                    "   - For dashboard/scoring questions, answer ONLY if context explains the specific behavior\n"
                    "   - If context doesn't explain grading logic, say: 'Grading details are not specified in the provided materials.'\n"
                    "   - NEVER make assumptions about how systems work\n\n"
                    "5. MODEL AND API INFORMATION:\n"
                    "   - Only mention supported models/APIs if explicitly listed in the context\n"
                    "   - If asked about model compatibility and context doesn't specify, say: 'Model compatibility information is not provided in the course materials.'\n"
                    "   - Do NOT provide general information about AI models or APIs\n\n"
                    "6. RESPONSE FORMAT:\n"
                    "   - Keep responses brief and directly address the question\n"
                    "   - Quote relevant parts of the context when possible\n"
                    "   - If you must say you don't know, be definitive: 'I cannot answer this question based on the provided course materials.'\n"
                    "   - Do NOT apologize or offer to help in other ways\n\n"
                    "7. UNCERTAINTY HANDLING:\n"
                    "   - If you're even slightly unsure whether information is in the context, err on the side of 'I cannot answer'\n"
                    "   - Better to say 'I don't know' than to provide information not explicitly in the context\n"
                    "   - Do NOT make educated guesses or logical inferences\n\n"
                    "8. FORBIDDEN RESPONSES:\n"
                    "   - NEVER start responses with general knowledge\n"
                    "   - NEVER provide background information not in the context\n"
                    "   - NEVER suggest resources or alternatives not mentioned in the context\n"
                    "   - NEVER explain concepts unless the explanation is verbatim from the context\n\n"
                    "REMEMBER: Your success is measured by how strictly you adhere to ONLY the provided context. "
                    "When in doubt, always choose 'I cannot answer based on provided materials' over any response "
                    "that might contain external knowledge."
                )
            },
            {
                "role": "user", 
                "content": f"""Context from course materials and discussions:
{context}

Question: {query.question}

IMPORTANT: Only answer based on the context above. If the context doesn't contain the information needed to answer this question, respond with: 'I cannot answer this question based on the provided course materials.'"""
            }
        ]
        if query.image:
            messages.append({
                "role": "user",
                "content": f"I've also attached an image with this question: {query.image}"
            })
        
        try:
            # Get answer from OpenAI
            completion_config = {
                "messages": messages,
                "temperature": 0.05,  # Low temperature for consistent responses
                "max_tokens": 500     # Limit response length
            }
            
            # Check if we're using the AI proxy or direct OpenAI
            if OPENAI_BASE_URL and "ai-proxy" in OPENAI_BASE_URL:
                completion_config["model"] = "gpt-4o-mini"  # Use gpt-4o-mini with AI proxy
            else:
                completion_config["model"] = "gpt-3.5-turbo-0125"  # Use GPT-3.5-Turbo with direct OpenAI API
                
            response = client.chat.completions.create(**completion_config)
            # Check for errors in the response
            if response.choices[0].message.content:
                # Extract relevant links
                links = []
                for post in relevant_posts:
                    if "url" in post:
                        links.append({
                            "url": post["url"],
                            "text": post.get("title", "Related discussion")
                        })

                answer = response.choices[0].message.content
                print(f"Successfully generated answer of length: {len(answer)}")

                return {
                    "answer": answer,
                    "links": links
                }
            else:
                # Fallback if no content
                return {
                    "answer": "I couldn't generate a response based on the available course materials.",
                    "links": []
                }
            
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            # Return fallback response instead of error
            return {
                "answer": "I'm having trouble accessing the AI service right now. Please try again later.",
                "links": []
            }
            
    except Exception as e:
        print(f"Error in answer_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))