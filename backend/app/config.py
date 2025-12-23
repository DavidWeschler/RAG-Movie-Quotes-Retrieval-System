"""
Configuration settings for the RAG system.

This module contains all configurable parameters for:
- Chunking strategy
- Retrieval parameters
- API settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # Default dimensions for text-embedding-3-small

# Chunking Strategy Configuration
# Reasoning: For short text data like movie quotes/reviews, smaller chunks work better
# 500 characters is enough to capture a complete thought while remaining focused
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks to maintain context continuity

# Retrieval Parameters
# Top-K: Number of most similar chunks to return
# Higher K = more context but potentially less relevant results
TOP_K = 5

# Similarity Threshold: Minimum cosine similarity score (0-1)
# Higher threshold = stricter matching, fewer but more relevant results
# 0.3 is a balanced threshold for semantic search
SIMILARITY_THRESHOLD = 0.3

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "movie_quotes"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
