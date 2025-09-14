import os
import io
from typing import List, Dict, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from fastapi import UploadFile

# Initialize Qdrant client
qdrant_client = QdrantClient(":memory:")
COLLECTION_NAME = "internship_documents"

# Initialize the sentence-transform embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#In memory storage for document metadata
metadata_db:Dict[str,Dict] = {}

def initialize_vector_db():
    """ensure the Qdrant collection exit"""
    print("Initializing vector database...")
    try:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE) 
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        print(f"Collection creation failed, it might already exit {e}")    

def extract_text_from_file(file:UploadFile) -> str:
    """Extract text from PDF or Text file."""
    content_type = file.content_type
    try:
        if content_type == "application/pdf":
            reader = PdfReader(io.BytesIO(file.file.read()))
            text = ""
        elif content_type == "text/plain":
            text = file.file.read().decode("utf-8")
            
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or Text file.")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
    finally:
        file.file.close()
    return text

def chunk_text(text:str, strategy:str) -> List[str]:
    """Chunk text using the specified strategy: 'fixed' or 'recursive'."""
    if strategy == "fixed":
        chunk_size = 500
        chunk_overlap = 50
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap   
        )
    elif strategy == "recursive":
        text_splitter = RecursiveCharacterTextSplitter( 
            chunk_size=500,
            chunk_overlap=50,
        )
    else:
        raise ValueError("Invalid chunking strategy. Choose 'fixed' or 'recursive'.")
    
    chunks = text_splitter.split_text(text)
    return chunks  
def create_embeddings(chunks:List[str]) -> List[List[float]]:
    """Generate embeddings for each text chunk using the sentence-transformers model."""
    embeddings = embedding_model.encode(chunks, show_progress_bar=True).tolist()
    return embeddings
        
def store_in_qdrant(document_id:str, chunks:List[str], embeddings:List[List[float]]):
    """Store text chunks and their embeddings in the Qdrant vector database."""
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = f"{document_id}_{i}"
        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "chunk_index": i,
                    "text_chunk": chunk
                }
            )
        )
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f" Successfully Stored {len(points)} points in Qdrant for document ID {document_id}.")  

def store_metadata(document_id:str, filename:str, chunking_strategy:str, num_chunks:int):
    """Store document metadata in the in-memory metadata_db."""
    metadata_db[document_id] = {
        "filename": filename,
        "chunking_strategy": chunking_strategy,
        "num_chunks": num_chunks
    }
    print(f"Metadata stored for document ID {document_id}.")