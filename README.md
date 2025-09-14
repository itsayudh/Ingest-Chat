Ingest and Chat (RAG-based AI Assistant)

This project implements a Retrieval-Augmented Generation (RAG) pipeline using FastAPI, Qdrant, Redis, and Google Gemini API.

It allows you to:

📥 Ingest PDF/Text documents into a vector database (Qdrant).

💬 Chat with your documents using Gemini (LLM).

🗂 Maintain chat history with Redis.

📅 Detect interview booking intents from user queries.

🚀 Features

Document Ingestion
Upload a PDF or text file → it’s chunked, embedded, and stored in Qdrant.

RAG Chat
Ask questions → relevant chunks are retrieved and passed to Gemini for context-aware answers.

Chat History
Conversations are stored in Redis and retrieved per session.

Booking Intent Detection
If a user requests an interview booking, the system extracts:

Name

Email

Date

Time

🛠️ Tech Stack

FastAPI
 – Web framework

Qdrant
 – Vector database for semantic search

Redis
 – Chat history storage

Google Gemini API
 – LLM for generation & intent detection

SentenceTransformers
 – Embedding model

 Installation & Setup
 1. Clone the repo
    git clone https://github.com/itsayudh/Ingest-and-Chat.git
    cd Ingest-and-Chat
    
 2. Create a virtual environment
    python -m venv venv
    source venv/bin/activate   # On Linux/Mac
    venv\Scripts\activate      # On Windows
  
 3. Install dependencies
    pip install -r requirements.txt

 4. Set environment variables
    Create a .env file (not pushed to GitHub) in the project root:
    GOOGLE_API_KEY=your_google_gemini_api_key
    REDIS_URL=redis://localhost:6379

 5. Run Redis (Docker recommended)
    docker run -d --name redis -p 6379:6379 redis

 6. Start the FastAPI server
    uvicorn main:app --reload

API Endpoints
1. Ingest Documents
POST /app/v1/ingest
Upload a document and specify chunking strategy (fixed or recursive).

2. Chat
POST /api/v1/chat
Send a query + session ID → get AI response with context.

3. Booking Intent
Integrated within chat – detects interview booking requests.

Project Structure
.
├── ingestion.py       # Handles document ingestion & embeddings
├── rag.py             # RAG pipeline & booking intent detection
├── main.py            # FastAPI app entrypoint
├── requirements.txt   # Dependencies
├── .env.example       # Example env vars
└── README.md          # Project documentation

