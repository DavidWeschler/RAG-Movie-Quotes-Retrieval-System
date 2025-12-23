"""
Dataset module for loading movie quotes from CSV.
Dataset Choice: Movie Quotes
"""

import csv
import os
from typing import List, Dict


def get_all_quotes() -> List[Dict]:
    """
    Load movie quotes from CSV file.
    Limits to first 303 rows to stay under 30,000 character limit.
    
    Returns:
        List of dictionaries containing quote data
    """
    quotes = []
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'movie_quotes.csv')
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            # Stop at line 303 to stay under 30,000 character limit
            if idx > 303:
                break
                
            quotes.append({
                "id": idx,
                "quote": row['quote'],
                "movie": row['movie'],
                "year": int(row['year']) if row['year'].isdigit() else 0,
                "type": row.get('type', 'movie')
            })
    
    return quotes


def format_quote_for_embedding(quote_data: dict) -> str:
    """
    Format a quote entry into a text string for embedding.
    
    Combines all relevant information into a single searchable text.
    This allows semantic search to find matches based on:
    - The quote text itself
    - The movie title
    - The year
    """
    return f"""Quote: "{quote_data['quote']}"
Movie: {quote_data['movie']} ({quote_data['year']})
Type: {quote_data['type']}"""


def get_formatted_quotes() -> List[Dict]:
    """Return all quotes formatted for embedding."""
    quotes = get_all_quotes()
    return [
        {
            "id": str(q["id"]),
            "text": format_quote_for_embedding(q),
            "metadata": {
                "movie": q["movie"],
                "year": q["year"],
                "type": q["type"],
                "original_quote": q["quote"]
            }
        }
        for q in quotes
    ]
