import os
import json
import time
import logging
from typing import Dict, Any, List

from app.config import NOT_AVAILABLE_RESPONSE
from app.modules.retriever import retrieve_relevant_chunks
from app.modules.prompt_builder import build_rag_prompt
from app.modules.llm_client import generate_llm_response

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluator")

def load_evaluation_set(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation() -> None:
    eval_set_path = os.path.join(os.path.dirname(__file__), "evaluation_set.json")
    if not os.path.exists(eval_set_path):
        logger.error(f"Evaluation set not found at {eval_set_path}")
        return

    logger.info(f"Loading evaluation set from {eval_set_path}...")
    eval_data = load_evaluation_set(eval_set_path)
    cases = eval_data.get("evaluation_cases", [])
    version = eval_data.get("version", "unknown")

    logger.info(f"Running evaluation for version {version} ({len(cases)} test cases)...")

    # Auto-index the evaluation document if present to populate vector database
    from app.modules.document_processor import extract_text_from_file, chunk_document_pages
    from app.modules.embeddings import generate_batch_embeddings
    from app.modules.vector_store import add_chunks_to_vector_store
    
    doc_name = "My project is a Personal Expense Tr.txt"
    doc_path = os.path.join(os.path.dirname(__file__), "..", "data", "uploads", doc_name)
    if os.path.exists(doc_path):
        logger.info(f"Found evaluation document '{doc_name}'. Indexing it before running evaluation...")
        try:
            pages = extract_text_from_file(doc_path)
            chunks = chunk_document_pages(pages, doc_name=doc_name)
            texts = [c["text"] for c in chunks]
            embs = generate_batch_embeddings(texts)
            add_chunks_to_vector_store(chunks, embs)
            logger.info("Successfully indexed evaluation document.")
        except Exception as e:
            logger.warning(f"Failed to auto-index evaluation document: {e}")

    results = []
    passed_cases = 0


    for case in cases:
        case_id = case["id"]
        category = case["category"]
        question = case["question"]
        expected_source = case.get("expected_source")
        expect_abstention = case.get("expect_abstention", False)
        keywords = case.get("target_answer_keywords", [])

        logger.info(f"[{category.upper()}] Case {case_id}: Question: '{question}'")

        start_time = time.time()
        # Step 1: Retrieval
        chunks = retrieve_relevant_chunks(question, top_k=3)
        retrieved_sources = [c["metadata"].get("source") for c in chunks if "metadata" in c]

        # Step 2: Generation
        prompt_payload = build_rag_prompt(question=question, chunks=chunks)
        answer = generate_llm_response(
            prompt_payload=prompt_payload,
            temperature=0.0,
            top_p=0.9,
            top_k=3,
            chunks=chunks
        )
        elapsed = time.time() - start_time

        # Check retrieval quality
        retrieval_ok = False
        if expected_source:
            retrieval_ok = any(expected_source == src for src in retrieved_sources)
        else:
            retrieval_ok = (len(chunks) == 0)

        # Check answer quality
        abstention_ok = False
        if expect_abstention:
            abstention_ok = (answer == NOT_AVAILABLE_RESPONSE)
        else:
            abstention_ok = (answer != NOT_AVAILABLE_RESPONSE)

        # Check keywords
        keyword_matches = []
        if keywords and not expect_abstention:
            for kw in keywords:
                if kw.lower() in answer.lower():
                    keyword_matches.append(kw)
            keyword_score = len(keyword_matches) / len(keywords)
            keyword_ok = keyword_score >= 0.5
        else:
            keyword_ok = True
            keyword_score = 1.0

        case_passed = retrieval_ok and abstention_ok and keyword_ok
        if case_passed:
            passed_cases += 1

        results.append({
            "id": case_id,
            "category": category,
            "question": question,
            "retrieved_sources": retrieved_sources,
            "answer": answer,
            "retrieval_ok": retrieval_ok,
            "abstention_ok": abstention_ok,
            "keyword_matches": keyword_matches,
            "keyword_score": keyword_score,
            "case_passed": case_passed,
            "latency_seconds": round(elapsed, 4)
        })

        logger.info(f"-> Passed: {case_passed} | Retrieval OK: {retrieval_ok} | Abstention OK: {abstention_ok} | Latency: {elapsed:.2f}s")

    success_rate = (passed_cases / len(cases)) * 100 if cases else 0.0
    logger.info(f"Evaluation completed. Passed {passed_cases}/{len(cases)} cases ({success_rate:.2f}%)")

    # Save results report
    report = {
        "version": version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "success_rate_percent": round(success_rate, 2)
        },
        "results": results
    }

    report_path = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {os.path.abspath(report_path)}")

if __name__ == "__main__":
    run_evaluation()
