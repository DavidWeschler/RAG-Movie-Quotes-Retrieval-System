"""
FastAPI Main Application

This is the main entry point for the RAG Retrieval System API.
It provides endpoints for:
- Initializing the vector store
- Searching for relevant context
- Getting system statistics
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from .vector_store import (
    initialize_vector_store,
    search_similar,
    get_collection_stats
)
from .config import TOP_K, SIMILARITY_THRESHOLD


# Create FastAPI app
app = FastAPI(
    title="RAG Retrieval System",
    description="A Retrieval-Augmented Generation system for searching movie quotes. "
                "This system demonstrates semantic search without LLM generation.",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., min_length=1, description="The search query")
    top_k: Optional[int] = Field(
        default=TOP_K,
        ge=1,
        le=20,
        description="Number of results to return"
    )
    similarity_threshold: Optional[float] = Field(
        default=SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1)"
    )


class SearchResult(BaseModel):
    """Individual search result."""
    id: str
    document: str
    metadata: Dict[str, Any]
    similarity_score: float
    distance: float


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str
    results: List[SearchResult]
    total_results: int
    parameters: Dict[str, Any]


class InitializeResponse(BaseModel):
    """Response model for initialization endpoint."""
    status: str
    message: str
    count: int


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    collection_name: str
    document_count: int
    persist_directory: str


# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "RAG Retrieval System is running",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    try:
        stats = get_collection_stats()
        return {
            "status": "healthy",
            "database": "connected",
            "documents_loaded": stats["document_count"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Initialize endpoint (supports both GET and POST for convenience)
@app.api_route("/initialize", methods=["GET", "POST"], response_model=InitializeResponse, tags=["Setup"])
async def initialize(force_rebuild: bool = False):
    """
    Initialize or reinitialize the vector store with movie quotes.
    
    This endpoint:
    1. Loads the movie quotes dataset
    2. Generates embeddings using OpenAI
    3. Stores embeddings in ChromaDB
    
    Args:
        force_rebuild: If True, delete existing data and rebuild
        
    Returns:
        Status of the initialization
    """
    try:
        result = initialize_vector_store(force_rebuild=force_rebuild)
        return InitializeResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize vector store: {str(e)}"
        )


# Search endpoint (POST for complex queries)
@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_post(request: SearchRequest):
    """
    Search for relevant movie quotes based on semantic similarity.
    
    This is the main RAG retrieval endpoint. It:
    1. Converts the query to an embedding
    2. Finds similar documents using cosine similarity
    3. Filters by similarity threshold
    4. Returns top-k most relevant results
    
    Args:
        request: SearchRequest with query and optional parameters
        
    Returns:
        SearchResponse with matching documents and scores
    """
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty or whitespace only"
            )
        
        results = search_similar(
            query=query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        return SearchResponse(
            query=query,
            results=[SearchResult(**r) for r in results],
            total_results=len(results),
            parameters={
                "top_k": request.top_k,
                "similarity_threshold": request.similarity_threshold
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# Search endpoint (GET for simple queries)
@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    query: str = Query(..., min_length=1, description="The search query"),
    top_k: int = Query(
        default=TOP_K,
        ge=1,
        le=20,
        description="Number of results to return"
    ),
    similarity_threshold: float = Query(
        default=SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )
):
    """
    Search for relevant movie quotes (GET version for simple queries).
    
    Same as POST /search but accepts query parameters.
    """
    request = SearchRequest(
        query=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )
    return await search_post(request)


# Stats endpoint
@app.get("/stats", response_model=StatsResponse, tags=["Info"])
async def get_stats():
    """
    Get statistics about the vector store.
    
    Returns:
        Collection name, document count, and storage location
    """
    try:
        stats = get_collection_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


# Configuration info endpoint
@app.get("/config", tags=["Info"])
async def get_config():
    """
    Get current configuration parameters.
    
    Useful for understanding system behavior and for frontend display.
    """
    from .config import (
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBEDDING_MODEL
    )
    
    return {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "default_top_k": TOP_K,
        "default_similarity_threshold": SIMILARITY_THRESHOLD
    }
