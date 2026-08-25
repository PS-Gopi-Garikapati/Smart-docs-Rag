"""
Query Routes Module.
FastAPI endpoint route for handling RAG user questions. Accepts dynamic generation
hyperparameters (Temperature, Top-P, Top-K) configured from the frontend.
"""

import os
import json
import re
import time
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.config import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, NOT_AVAILABLE_RESPONSE
from app.modules.retriever import retrieve_relevant_chunks
from app.modules.prompt_builder import build_rag_prompt
from app.modules.llm_client import generate_llm_response

logger = logging.getLogger(__name__)

def compare_filenames(file1: str, file2: str) -> bool:
    if not file1 or not file2:
        return False
    norm1 = file1.lower().replace("_", "").replace("-", "").replace(" ", "")
    norm2 = file2.lower().replace("_", "").replace("-", "").replace(" ", "")
    return norm1 == norm2


def is_abstention_response(answer: str) -> bool:
    if not answer:
        return True
    ans_lower = answer.lower()
    fallback_lower = NOT_AVAILABLE_RESPONSE.lower()
    return (
        fallback_lower in ans_lower or
        "don't have relevant answer" in ans_lower or
        "no relevant document" in ans_lower or
        "do not have relevant" in ans_lower or
        "cannot answer" in ans_lower or
        "unable to answer" in ans_lower
    )

router = APIRouter(prefix="/api", tags=["RAG Query Engine"])

# Load evaluation cases on startup
EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "evaluation_set.json")
EVAL_CASES = []
if os.path.exists(EVAL_SET_PATH):
    try:
        with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            EVAL_CASES = eval_data.get("evaluation_cases", [])
            logger.info(f"Loaded {len(EVAL_CASES)} evaluation cases for dynamic response checking.")
    except Exception as e:
        logger.error(f"Error loading evaluation cases: {e}")


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
    evaluation_results: Optional[Dict[str, Any]] = None


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
        from app.modules.vector_store import sync_vector_store_with_uploads
        sync_vector_store_with_uploads()
        
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
        retrieved_sources_raw = []
        for idx, c in enumerate(chunks, 1):
            meta = c.get("metadata", {})
            src = meta.get("source", "Unknown Document")
            retrieved_sources_raw.append(src)
            citations.append({
                "citation_id": idx,
                "source": src,
                "page": meta.get("page", 1),
                "similarity_score": c.get("similarity", 0.0),
                "snippet": c.get("text", "")[:250] + ("..." if len(c.get("text", "")) > 250 else "")
            })

        # Dynamic correctness evaluation
        matched_case = None
        norm_q = question.lower().rstrip("?. ")
        for case in EVAL_CASES:
            case_q = case.get("question", "").lower().rstrip("?. ")
            if norm_q == case_q:
                matched_case = case
                break

        expected_source = matched_case.get("expected_source") if matched_case else None
        expect_abstention = matched_case.get("expect_abstention", False) if matched_case else False
        keywords = matched_case.get("target_answer_keywords", []) if matched_case else []

        # Run evaluation using our new module
        from app.modules.evaluator import evaluate_rag_response, check_keyword_in_text
        eval_results = evaluate_rag_response(
            question=question,
            answer=answer,
            chunks=chunks,
            expected_keywords=keywords,
            expect_abstention=expect_abstention
        )

        if matched_case:
            if expected_source:
                retrieval_ok = any(compare_filenames(expected_source, src) for src in retrieved_sources_raw)
            else:
                if matched_case.get("category") == "adversarial":
                    retrieval_ok = True
                else:
                    retrieval_ok = (len(chunks) == 0)
            
            # Check answer quality (abstention)
            if expect_abstention:
                abstention_ok = is_abstention_response(answer)
            else:
                abstention_ok = not is_abstention_response(answer)

            # Check keyword match
            if keywords and not expect_abstention:
                keyword_matches = [kw for kw in keywords if check_keyword_in_text(kw, answer)]
                keyword_score = len(keyword_matches) / len(keywords)
                keyword_ok = keyword_score >= 0.5
            else:
                keyword_matches = []
                keyword_score = 1.0
                keyword_ok = True

            case_passed = retrieval_ok and abstention_ok and keyword_ok

            eval_results["has_ground_truth"] = True
            eval_results["case_passed"] = case_passed
            eval_results["retrieval_ok"] = retrieval_ok
            eval_results["abstention_ok"] = abstention_ok
            eval_results["keyword_ok"] = keyword_ok
            eval_results["keyword_matches"] = keyword_matches
            eval_results["keyword_score"] = keyword_score
            eval_results["expected_source"] = expected_source
            eval_results["expect_abstention"] = expect_abstention
            eval_results["precision"] = 1.0 if retrieval_ok else 0.0
            eval_results["recall"] = 1.0 if retrieval_ok else 0.0
        else:
            eval_results["has_ground_truth"] = False
            eval_results["grounding_score"] = eval_results.get("faithfulness", 0.0)
            eval_results["grounding_ok"] = eval_results.get("faithfulness", 0.0) >= 0.25

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
            },
            evaluation_results=eval_results
        )

    except Exception as e:
        logger.error(f"Error handling query request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating the answer: {str(e)}"
        )


@router.post("/evaluate/run", status_code=status.HTTP_200_OK)
async def trigger_evaluation() -> Dict[str, Any]:
    """
    Triggers the benchmark evaluation runner against evaluation_set.json.
    """
    try:
        from app.evaluate import run_evaluation
        run_evaluation()
        
        report_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "evaluation_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Evaluation completed but report file was not generated."
            )
    except Exception as e:
        logger.error(f"Error running evaluation suite: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run evaluation: {str(e)}"
        )


@router.get("/evaluate/report", status_code=status.HTTP_200_OK)
async def get_evaluation_report() -> Dict[str, Any]:
    """
    Retrieves the latest generated evaluation report.
    """
    report_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "evaluation_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read evaluation report: {str(e)}"
            )
    else:
        return {
            "status": "unevaluated",
            "summary": {
                "total_cases": 0,
                "passed_cases": 0,
                "success_rate_percent": 0.0
            },
            "results": []
        }

