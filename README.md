Internship RAG Backend
This is a backend application built with FastAPI that provides two REST APIs for a Retrieval-Augmented Generation (RAG) system. The application is designed to ingest documents and then use that knowledge to answer user queries in a conversational manner, including a special feature for booking interviews.

The project follows a modular and clean code structure, adhering to industry standards for typing and annotations.

Features
Document Ingestion API:
Accepts .pdf and .txt file uploads.
Extracts text from documents.
Supports two selectable chunking strategies for text processing.
Generates and stores text embeddings in a vector database (Qdrant).
Stores document metadata in an in-memory database (simulating a SQL/NoSQL DB).

Conversational RAG API:

Implements a custom RAG pipeline (no RetrievalQAChain from libraries).
Uses Redis for efficient management of multi-turn chat memory.
Handles multi-turn queries by providing context from previous conversations.
Supports natural language-based interview booking, extracting key information like name, email, date, and time.
Stores booking information from the conversation.

File Structure
<img width="816" height="273" alt="image" src="https://github.com/user-attachments/assets/4a10ed59-fc7d-4dce-8c66-2cfd2e2bb30a" />


Getting Started

Prerequisites
Python 3.9+

Docker (for running the Redis container)

A Google API Key

Step 1: Clone the Repository
Clone this repository to your local machine using Git.

git clone [https://github.com/your-username/Internship-RAG-App.git](https://github.com/itsayudh/Ingest-Chat.git)
cd Ingest-Chat.git

Step 2: Set up Environment Variables
Create a file named .env in the root directory of your project and add your Google API key. Do not commit this file to Git.

GOOGLE_API_KEY="your-google-api-key-here"

Step 3: Install Dependencies
Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

Step 4: Run Redis
Start a Redis container using Docker. This is essential for the chat memory functionality.

docker run --name my-redis -d -p 6379:6379 redis

Step 5: Run the Backend Application
Start the FastAPI application with uvicorn. The --reload flag will automatically restart the server on code changes.

uvicorn main:app --reload

Your backend server should now be running locally at http://127.0.0.1:8000.

API Endpoints
The application exposes the following REST API endpoints:

1. Document Ingestion API
Endpoint: POST /api/v1/ingest

Description: Uploads a .pdf or .txt file, processes it, and stores the embeddings in the vector database.

Form Data:

file: The document file to be uploaded.

chunking_strategy: A string, either "fixed" or "recursive".

2. Conversational RAG API
Endpoint: POST /api/v1/chat

Description: Receives a user query and returns a generated response based on the ingested documents. It also handles chat history and special commands for interview booking.

Request Body (JSON):

session_id: A unique string to identify the chat session.

user_query: The user's message.

How to Test
You can use a tool like Postman, Insomnia, or curl to send requests to the API endpoints and test their functionality.

# Example curl command for the Ingestion API
curl -X POST "[http://127.0.0.1:8000/api/v1/ingest?chunking_strategy=recursive](http://127.0.0.1:8000/api/v1/ingest?chunking_strategy=recursive)" \
-H "accept: application/json" -H "Content-Type: multipart/form-data" \
-F "file=@your-document.pdf"
