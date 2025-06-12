# TDS Virtual TA

A Virtual Teaching Assistant API for IIT Madras Tools in Data Science course that automatically answers student questions based on course content and Discourse discussions.

## Features

- 🤖 **AI-Powered Responses**: Uses advanced language models to provide contextual answers
- 📚 **Course Content Integration**: Searches through TDS course materials
- 💬 **Discourse Integration**: Includes relevant discussions from course forums
- 🖼️ **Image Processing**: Handles base64 image attachments for visual questions
- ⚡ **Fast Response**: Optimized for <30 second response times
- 🔍 **Hybrid Search**: Combines semantic and keyword search for better accuracy

## API Usage

### Endpoint
```
POST /api/
```

### Request Format
```json
{
  "question": "Your question here",
  "image": "base64_encoded_image (optional)"
}
```

### Response Format
```json
{
  "answer": "Detailed answer based on course content",
  "links": [
    {
      "url": "https://relevant-link.com",
      "text": "Description of the linked content"
    }
  ]
}
```

### Example Usage

```bash
# Text-only question
curl "https://your-deployment.vercel.app/api/" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I read a CSV file in pandas?"}'

# Question with image
curl "https://your-deployment.vercel.app/api/" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What does this error mean?\", \"image\": \"$(base64 -w0 error_screenshot.png)\"}"
```

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/tds-virtual-ta
cd tds-virtual-ta
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Scrape Data
```bash
# Scrape course content
python -m scraper.content_scraper

# Scrape Discourse posts (BONUS feature)
python -m scraper.discourse_scraper --start-date 2025-01-01 --end-date 2025-04-14
```

### 4. Run Locally
```bash
uvicorn main:app --reload --port 8000
```

### 5. Deploy to Vercel
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

## Project Structure

```
tds-virtual-ta/
├── LICENSE                    # MIT License
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── vercel.json               # Vercel deployment config
├── main.py                   # FastAPI application
├── scraper/
│   ├── __init__.py
│   ├── discourse_scraper.py  # Discourse scraping (BONUS)
│   └── content_scraper.py    # Course content scraping  
├── utils/
│   ├── __init__.py
│   ├── search.py            # Knowledge base & search
│   └── ai_proxy.py          # AI response generation
└── data/
    ├── course_content.json   # Scraped course content
    ├── discourse_posts.json  # Scraped discussions
    └── embeddings_cache.pkl  # Cached embeddings
```

## Bonus Features

### ✅ Discourse Scraping Script (1 point)
The `scraper/discourse_scraper.py` script can scrape TDS Discourse posts within any date range:

```bash
python -m scraper.discourse_scraper --start-date 2025-01-01 --end-date 2025-04-14 --output data/posts.json
```

### ✅ Production-Ready Deployment (2 points)
- Deployed on Vercel with minimal configuration
- Optimized for performance and reliability
- Proper error handling and fallbacks
- Ready for student use

## Technical Details

### Search System
- **Semantic Search**: Uses sentence-transformers for contextual understanding
- **Keyword Search**: TF-IDF based matching for exact terms
- **Hybrid Ranking**: Combines both approaches for optimal results

### AI Integration
- Uses the free aiproxy.sanand.workers.dev service
- Implements proper prompt engineering for educational context
- Includes fallback responses for reliability

### Performance Optimizations
- Caches embeddings to avoid recomputation
- Async processing for faster response times
- Efficient document retrieval and ranking

## Development

### Testing the API
```bash
# Run tests
python -m pytest tests/

# Manual testing
python test_api.py
```

### Code Quality
```bash
# Format code
black .

# Lint code  
flake8 .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IIT Madras TDS Course Team
- AI Proxy service by Sanand
- Open source libraries: FastAPI, sentence-transformers, scikit-learn

---

**Created for the TDS Virtual TA Project**  
*Helping students succeed with AI-powered assistance* 🎓