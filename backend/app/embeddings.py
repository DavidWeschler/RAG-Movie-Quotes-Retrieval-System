"""
Embeddings module for generating text embeddings using OpenAI.
Embedding Model Choice: text-embedding-3-small
"""

from openai import OpenAI
from typing import List
from .config import OPENAI_API_KEY, EMBEDDING_MODEL


def get_openai_client():
    """Get OpenAI client instance."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text string.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    client = get_openai_client()
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    
    return response.data[0].embedding


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a single API call.
    
    More efficient than calling generate_embedding multiple times.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    client = get_openai_client()
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    
    # Sort by index to maintain order
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]
