"""
Retriever Module.
Coordinates text embedding generation for user questions and executes semantic
vector retrieval against the ChromaDB vector database.
"""

import logging
from typing import List, Dict, Any
from app.modules.embeddings import generate_text_embedding
from app.modules.vector_store import query_vector_store
from app.config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(question: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Purpose:
    Retrieves the top-K most semantically relevant text chunks for a user's question.

    Parameters:
    question (str): The natural language query entered by the user.
    top_k (int): The number of candidate chunks to retrieve. Configurable from frontend.

    Returns:
    List[Dict[str, Any]]: List of retrieved document chunk dictionaries with text, metadata, and score.

    Raises:
    ValueError: If question string is empty.
    RuntimeError: If vector store retrieval fails.
    """
    if not question or not question.strip():
        raise ValueError("Question string cannot be empty.")

    # Enforce valid bounds for top_k parameter
    k = max(1, min(int(top_k), 50))

    logger.info(f"Retrieving top {k} relevant chunks for question: '{question}'")

    try:
        # Step 1: Generate query embedding using Sentence Transformers
        query_embedding = generate_text_embedding(question)

        # Step 2: Query ChromaDB vector database with dense embedding
        matched_chunks = query_vector_store(query_embedding=query_embedding, top_k=k)

        logger.info(f"Retrieved {len(matched_chunks)} chunks from vector store.")
        return matched_chunks

    except Exception as e:
        logger.error(f"Error during context retrieval: {e}")
        raise RuntimeError(f"Context retrieval failed: {str(e)}")

