"""
Vector Store module using ChromaDB.
This module handles:
- Initializing the vector store with movie quotes
- Storing and retrieving embeddings
- Searching for similar documents based on a query
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os

from .config import (
    CHROMA_PERSIST_DIRECTORY,
    COLLECTION_NAME,
    TOP_K,
    SIMILARITY_THRESHOLD
)
from .embeddings import generate_embedding, generate_embeddings_batch
from .dataset import get_formatted_quotes


def get_chroma_client():
    """
    Get ChromaDB client with persistent storage.
    
    Uses persistent storage so embeddings survive between sessions.
    This avoids regenerating embeddings (and incurring API costs) on restart.
    """
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIRECTORY,
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def get_or_create_collection():
    """
    Get existing collection or create a new one.
    
    Uses cosine similarity for distance metric - standard for text embeddings.
    """
    client = get_chroma_client()
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    return collection


def initialize_vector_store(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Initialize the vector store with movie quotes.
    
    Args:
        force_rebuild: If True, delete existing data and rebuild from scratch
        
    Returns:
        Dictionary with status information
    """
    collection = get_or_create_collection()
    
    # Check if collection already has data
    existing_count = collection.count()
    
    if existing_count > 0 and not force_rebuild:
        return {
            "status": "exists",
            "message": f"Collection already initialized with {existing_count} documents",
            "count": existing_count
        }
    
    # If force rebuild, delete existing collection
    if force_rebuild and existing_count > 0:
        client = get_chroma_client()
        client.delete_collection(COLLECTION_NAME)
        collection = get_or_create_collection()
    
    # Get formatted quotes
    quotes = get_formatted_quotes()
    
    # Prepare data for insertion
    ids = [q["id"] for q in quotes]
    texts = [q["text"] for q in quotes]
    metadatas = [q["metadata"] for q in quotes]
    
    # Generate embeddings in batch (more efficient)
    print("Generating embeddings...")
    embeddings = generate_embeddings_batch(texts)
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    return {
        "status": "created",
        "message": f"Successfully initialized collection with {len(quotes)} documents",
        "count": len(quotes)
    }


def search_similar(
    query: str,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar documents based on a query.
    
    Args:
        query: The search query text
        top_k: Number of results to return (default from config)
        similarity_threshold: Minimum similarity score (default from config)
        
    Returns:
        List of matching documents with scores and metadata
        
    Retrieval Strategy:
    - First, embed the query using the same model as documents
    - Search using cosine similarity
    - Filter by similarity threshold
    - Return top-k most similar results
    """
    if top_k is None:
        top_k = TOP_K
    if similarity_threshold is None:
        similarity_threshold = SIMILARITY_THRESHOLD
    
    collection = get_or_create_collection()
    
    # Check if collection has data
    if collection.count() == 0:
        return []
    
    # Generate embedding for query
    query_embedding = generate_embedding(query)
    
    # Query the collection
    # Note: ChromaDB returns distance, not similarity
    # For cosine distance: similarity = 1 - distance
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    processed_results = []
    
    if results and results['ids'] and len(results['ids']) > 0:
        for i, doc_id in enumerate(results['ids'][0]):
            # Convert distance to similarity score
            distance = results['distances'][0][i]
            similarity = 1 - distance  # Cosine similarity = 1 - cosine distance
            
            # Apply similarity threshold
            if similarity >= similarity_threshold:
                processed_results.append({
                    "id": doc_id,
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity_score": round(similarity, 4),
                    "distance": round(distance, 4)
                })
    
    return processed_results


def get_collection_stats() -> Dict[str, Any]:
    """Get statistics about the current collection."""
    collection = get_or_create_collection()
    
    return {
        "collection_name": COLLECTION_NAME,
        "document_count": collection.count(),
        "persist_directory": CHROMA_PERSIST_DIRECTORY
    }
