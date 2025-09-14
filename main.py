from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from dotenv import load_dotenv
import os
import uuid
import json
from pydantic import BaseModel
from ingestion import extract_text_from_file, chunk_text, create_embeddings, store_in_qdrant, store_metadata, initialize_vector_db
from rag import run_rag_pipeline, detect_booking_intent, save_chat_message

load_dotenv()

app = FastAPI(
    title="RAG Application",
    description="A FastAPI application for Retrieval-Augmented Generation (RAG) with file ingestion and chat capabilities.",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Application! Server is running."}

@app.on_event("startup")
async def starup_event():
    """Initializes the vector database on startup."""
    initialize_vector_db()

@app.post("/app/v1/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(..., description="the chunking strategy to use:'fixed' or 'recursive '")
):
    """Uploads a documents (PDF or Text), extracts text,chunks it, generates embeddings, 
    and store them in vector database with associated metadata."""
    # Validate chunking strategy
    if chunking_strategy not in ['fixed','recursive' ]:
        raise HTTPException(
            status_code= 400,
            detail="Invalid chunking strategy. Choose 'fixed' or 'recursive'."
        )
    
    #Extract text from the uploaded file
    try:
        
        raw_text = extract_text_from_file(file)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )    
    #apply chunking strategy
    text_chunks = chunk_text(raw_text, chunking_strategy)

    #generate embedings for each chumks
    embeddings = create_embeddings(text_chunks)

    #generate a unique documnents id and store everythings
    document_id = str(uuid.uuid4())
    store_in_qdrant(document_id, text_chunks, embeddings)
    store_metadata(document_id, file.filename,chunking_strategy,len(text_chunks))

    return{
        "status":"success",
        "message":f"Document '{file.filename}' ingested sucessfully.",
        "document_id": document_id,
        "chunking_strategy": chunking_strategy,
        "num_chunks": len(text_chunks)
    }
class ChatRequest(BaseModel):
    session_id:str
    user_query:str

#conversational RAG API endopoint
@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """Handles a user's conversationalquery,detects booking intent,
    and  use RAG to provide a relevant response"""
    user_query = request.user_query
    session_id = request.session_id
    # checking the interview booking intent
    booking_result = detect_booking_intent(user_query)

    if isinstance(booking_result,dict) and booking_result:
        print(f"Interview booking detected and details extracted: {json.dumps(booking_result, indent=2)}")

        save_chat_message(session_id, "user", user_query)
        response_text = "Thank you! I have Received your booking details.We will contact you soon to confirm the interview"
        save_chat_message(session_id, "assistant", response_text)

        return{
            "status":"booking_success",
            "message":response_text,
            "booking_details":booking_result
        }
    elif isinstance(booking_result, str):
        #the booking intent is detected, busome info is missing
        save_chat_message(session_id, "user", user_query)
        save_chat_message(session_id, "assistant", booking_result)
        return{
            "status":"booking_incomplete",
            "message":booking_result
        }
    #If no booking intent is detected, run  RAG pipeline
    save_chat_message(session_id, "user", user_query)
    response_text = run_rag_pipeline(user_query,session_id)
    save_chat_message(session_id, "assistant", response_text)

    return{
        "status":"success",
        "message":response_text
    }