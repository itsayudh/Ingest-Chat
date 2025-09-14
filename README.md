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
