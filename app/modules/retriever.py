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

        # Direct Keyword Match Fallback:
        # If the best result is below 0.35 similarity, search chunks for direct keyword matches (especially for CSV IDs like 1007)
        if not matched_chunks or matched_chunks[0].get("similarity", 0) < 0.35:
            from app.modules.vector_store import get_vector_store, PersistentJsonVectorStore
            store = get_vector_store()
            chunks_to_search = []
            if isinstance(store, PersistentJsonVectorStore):
                chunks_to_search = store.chunks
            else:
                try:
                    all_records = store.get(include=["documents", "metadatas"])
                    if all_records and all_records.get("documents"):
                        chunks_to_search = [
                            {"text": doc, "metadata": meta}
                            for doc, meta in zip(all_records["documents"], all_records.get("metadatas", []))
                        ]
                except Exception:
                    pass

            if chunks_to_search:
                # Find numbers or codes in query (e.g. 1007)
                stop_words = {
                    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
                    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
                    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
                    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
                    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
                    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
                    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
                    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
                    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
                    "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours",
                    "tell", "show", "give", "bot", "answer"
                }
                q_words = [w.strip().lower() for w in question.replace("=", " ").replace(":", " ").split() if len(w.strip()) > 1 and w.strip().lower() not in stop_words]
                if q_words:
                    direct_matches = []
                    for item in chunks_to_search:
                        text_lower = item["text"].lower()
                        matches = 0
                        for word in q_words:
                            if word in text_lower:
                                matches += 1
                        if matches > 0:
                            direct_matches.append({
                                "text": item["text"],
                                "metadata": item["metadata"],
                                "distance": 0.0,
                                "similarity": 0.9 + (0.09 * (matches / len(q_words)))
                            })
                    if direct_matches:
                        direct_matches.sort(key=lambda x: x["similarity"], reverse=True)
                        matched_chunks = direct_matches[:k]

        logger.info(f"Retrieved {len(matched_chunks)} chunks from vector store.")
        return matched_chunks

    except Exception as e:
        logger.error(f"Error during context retrieval: {e}")
        raise RuntimeError(f"Context retrieval failed: {str(e)}")


