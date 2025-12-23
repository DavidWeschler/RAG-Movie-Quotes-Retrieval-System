"""
Chunking Strategy Module
This module provides functions to split long text documents into smaller,
overlapping chunks suitable for embedding and retrieval.
"""

from typing import List, Dict, Any
from .config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split a long text into overlapping chunks.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
        
    Example:
        text = "ABCDEFGHIJ", chunk_size=4, overlap=1
        Result: ["ABCD", "DEFG", "GHIJ"]
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = end - overlap
        
        # Prevent infinite loop if overlap >= chunk_size
        if overlap >= chunk_size:
            start = end
    
    return chunks


def chunk_document(
    document: Dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Chunk a document while preserving metadata.
    
    Each chunk inherits the document's metadata plus chunk-specific info.
    
    Args:
        document: Dict with 'id', 'text', and 'metadata' keys
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries
    """
    text = document.get("text", "")
    base_id = document.get("id", "unknown")
    metadata = document.get("metadata", {})
    
    chunks = split_text_into_chunks(text, chunk_size, overlap)
    
    result = []
    for i, chunk_text in enumerate(chunks):
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_index"] = i
        chunk_metadata["total_chunks"] = len(chunks)
        chunk_metadata["original_doc_id"] = base_id
        
        result.append({
            "id": f"{base_id}_chunk_{i}",
            "text": chunk_text,
            "metadata": chunk_metadata
        })
    
    return result


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of document dictionaries
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks
        
    Returns:
        Flattened list of all chunks from all documents
    """
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
    
    return all_chunks
