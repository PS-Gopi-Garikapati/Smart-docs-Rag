"""
Query Routes Module.
FastAPI endpoint route for handling RAG user questions. Accepts dynamic generation
hyperparameters (Temperature, Top-P, Top-K) configured from the frontend.
"""

import time
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.config import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K
from app.modules.retriever import retrieve_relevant_chunks
from app.modules.prompt_builder import build_rag_prompt
from app.modules.llm_client import generate_llm_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG Query Engine"])


class QueryRequest(BaseModel):
    """
    Purpose:
    Pydantic schema model for RAG question query requests.
    Enforces frontend parameter configuration boundaries.

    Responsibilities:
    - Validate query request body elements.
    - Bound hyperparameter values.
    """
    question: str = Field(..., description="User question string", min_length=1)
    temperature: Optional[float] = Field(
        DEFAULT_TEMPERATURE,
        ge=0.0,
        le=1.0,
        description="LLM generation temperature (0 to 1)"
    )
    top_p: Optional[float] = Field(
        DEFAULT_TOP_P,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold (0 to 1)"
    )
    top_k: Optional[int] = Field(
        DEFAULT_TOP_K,
        ge=1,
        le=50,
        description="Number of document chunks to retrieve (1 to 50)"
    )



class QueryResponse(BaseModel):
    """
    Purpose:
    Pydantic schema model for RAG answer responses.
    """
    status: str
    question: str
    answer: str
    retrieved_sources: List[Dict[str, Any]]
    execution_time_seconds: float
    parameters_used: Dict[str, Any]


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def ask_question(request: QueryRequest) -> QueryResponse:
    """
    Purpose:
    Executes Retrieval-Augmented Generation (RAG) pipeline for user questions.

    Pipeline Steps:
    1. Validates input question and hyperparameters.
    2. Generates query vector embedding and retrieves Top-K relevant document chunks from ChromaDB.
    3. Builds context-grounded RAG prompt with strict fallback instructions.
    4. Executes LLM answer generation using configured Temperature, Top-P, and Top-K.
    5. Returns answer along with retrieved citation sources and latency timing.

    Parameters:
    request (QueryRequest): The RAG query request payload.

    Returns:
    QueryResponse: Generated answer along with context sources and metadata.

    Raises:
    HTTPException: 400 Bad Request if question is empty, 500 on execution failures.
    """
    start_time = time.time()
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty."
        )

    # Use defaults if optional parameters are null
    temp = request.temperature if request.temperature is not None else DEFAULT_TEMPERATURE
    top_p = request.top_p if request.top_p is not None else DEFAULT_TOP_P
    top_k = request.top_k if request.top_k is not None else DEFAULT_TOP_K
    logger.info(f"Received query request: '{question}' | Temp={temp}, Top-P={top_p}, Top-K={top_k}")

    try:
        # Step 1: Retrieve relevant context chunks using vector similarity search
        chunks = retrieve_relevant_chunks(question=question, top_k=top_k)

        # Step 2: Build prompt with retrieved context
        prompt_payload = build_rag_prompt(question=question, chunks=chunks)

        # Step 3: Call LLM engine with configured parameters
        answer = generate_llm_response(
            prompt_payload=prompt_payload,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            chunks=chunks
        )

        elapsed = round(time.time() - start_time, 3)

        # Format retrieved citations for frontend display
        citations = []
        for idx, c in enumerate(chunks, 1):
            meta = c.get("metadata", {})
            citations.append({
                "citation_id": idx,
                "source": meta.get("source", "Unknown Document"),
                "page": meta.get("page", 1),
                "similarity_score": c.get("similarity", 0.0),
                "snippet": c.get("text", "")[:250] + ("..." if len(c.get("text", "")) > 250 else "")
            })

        return QueryResponse(
            status="success",
            question=question,
            answer=answer,
            retrieved_sources=citations,
            execution_time_seconds=elapsed,
            parameters_used={
                "temperature": temp,
                "top_p": top_p,
                "top_k": top_k
            }
        )

    except Exception as e:
        logger.error(f"Error handling query request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating the answer: {str(e)}"
        )

