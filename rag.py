import os
import json
import redis
import google.generativeai as genai
from typing import List, Dict, Union
from pydantic import BaseModel, Field, ValidationError
from qdrant_client import QdrantClient
from qdrant_client.http.models import models
from ingestion import embedding_model, COLLECTION_NAME

# Load environment variables
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))
    qdrant_client = QdrantClient(":memory:")
except Exception as e:
    print(f"Failed to initialize RAG components: {e}")
    

class BookingDetails(BaseModel):
    name: str = Field(..., description="The full name of the person booking the interview.")
    email: str = Field(..., description="The email address of the person.")
    date: str = Field(..., description="The preferred date for the interview, in a clear format.")
    time: str = Field(..., description="The preferred time for the interview, in a clear format.")

def search_qdrant(query: str, top_k: int = 3) -> List[str]:
    """
    Searches the Qdrant vector database for the most relevant document chunks.
    
    Args:
        query (str): The user's query.
        top_k (int): The number of relevant chunks to retrieve.

    Returns:
        List[str]: A list of text chunks retrieved from the database.
    """
    try:
        query_embedding = embedding_model.encode(query).tolist()
        
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k
        )
        
        relevant_chunks = [point.payload.get("chunk_text") for point in search_result if point.payload]
        return relevant_chunks
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        return []

def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """
    Retrieves chat history from Redis for a given session.
    Returns a list of messages.
    """
    history_key = f"chat_history:{session_id}"
    try:
        history = redis_client.lrange(history_key, 0, -1)
        return [json.loads(message.decode('utf-8')) for message in history]
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error: {e}")
        return []
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def save_chat_message(session_id: str, role: str, message: str):
    """Saves a message to the Redis chat history."""
    history_key = f"chat_history:{session_id}"
    try:
        redis_client.rpush(history_key, json.dumps({"role": role, "message": message}))
        redis_client.expire(history_key, 3600)  # Expire history after 1 hour
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error: {e}")
    except Exception as e:
        print(f"Error saving chat history: {e}")

def run_rag_pipeline(user_query: str, session_id: str) -> str:
    """
    Runs the full RAG pipeline: retrieval, augmentation, and generation.
    """
    print("Running custom RAG pipeline...")
    # 1. Retrieval
    relevant_chunks = search_qdrant(user_query)
    
    # 2. Augmentation
    chat_history = get_chat_history(session_id)
    context = "\n".join(relevant_chunks)
    
    prompt = f"""
    You are an AI assistant that answers questions based on the provided documents and the chat history. 
    Be concise, helpful, and polite. If the answer is not in the documents, state that you cannot answer.

    ---
    Chat History:
    {json.dumps(chat_history, indent=2)}

    ---
    Documents:
    {context}

    ---
    User Query:
    {user_query}
    """
    
    # 3. Generation
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return "I am unable to generate a response at this time. Please try again later."


def detect_booking_intent(user_query: str) -> Union[Dict, str]:
    """
    Uses the LLM to detect interview booking intent and extract details.
    
    Returns:
        Union[Dict, str]: A dictionary of booking details or a string message.
    """
    prompt = f"""
    The user wants to book an interview. Extract the name, email, date, and time.
    If any information is missing, do not guess, and state what is missing.
    Format your response as a single JSON object with the following schema:
    
    {{
        "name": "...",
        "email": "...",
        "date": "...",
        "time": "..."
    }}

    If the user's request is not for an interview booking, return an empty JSON object {{}}.
    Do not add any additional text or explanation.

    User's request: {user_query}
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        # Parse the JSON response
        response_text = response.text.strip().replace('```json', '').replace('```', '')
        booking_data = json.loads(response_text)
        
        # Validate the data with Pydantic
        booking_details = BookingDetails(**booking_data)
        
        # Check for missing fields
        missing_fields = []
        if not booking_details.name: missing_fields.append("name")
        if not booking_details.email: missing_fields.append("email")
        if not booking_details.date: missing_fields.append("date")
        if not booking_details.time: missing_fields.append("time")
        
        if missing_fields:
            return f"I am missing the following information to book your interview: {', '.join(missing_fields)}. Can you please provide them?"
            
        return booking_details.model_dump()

    except ValidationError:
        return "I couldn't extract all the required information to book the interview. Please provide your full name, email, preferred date, and time."
    except json.JSONDecodeError:
        return {} # Not a booking request
    except Exception as e:
        print(f"Error in booking intent detection: {e}")
        return "I am unable to process your booking request at this time."
