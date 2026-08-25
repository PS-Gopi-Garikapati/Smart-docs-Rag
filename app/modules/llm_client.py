"""
LLM Client Module.
Handles communication with local Ollama (Llama3) LLM,
dynamically applying hyperparameter configurations (Temperature, Top-P, Top-K)
passed from the frontend. Includes a local extractive fallback engine when Ollama is unavailable.
"""

import os
import re
import logging
from typing import Dict, Any, List
from app.config import OLLAMA_HOST, OLLAMA_MODEL_NAME, NOT_AVAILABLE_RESPONSE, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def get_word_root(w: str) -> str:
    """
    Normalizes a word to its root stem by removing common plurals and verb suffixes.
    """
    w = w.lower()
    if w.endswith("ies"):
        w = w[:-3] + "i"
    elif w.endswith("es") and not w.endswith("ces") and not w.endswith("ses"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        w = w[:-1]
    elif w.endswith("ing"):
        w = w[:-3]
    elif w.endswith("ed"):
        w = w[:-2]
    return w[:4] if len(w) >= 4 else w


def generate_llm_response(
    prompt_payload: Dict[str, str],
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    chunks: List[Dict[str, Any]] = None
) -> str:
    """
    Purpose:
    Generates a context-grounded answer using local Ollama model or fallback engine.

    Parameters:
    prompt_payload (Dict[str, str]): Dictionary containing 'system_instruction' and 'user_content'.
    temperature (float): Controls response randomness (0.0 to 1.0). Configured from frontend slider.
    top_p (float): Nucleus sampling threshold (0.0 to 1.0). Configured from frontend slider.
    top_k (int): Top-K sampling filter (1 to 100). Configured from frontend input.
    chunks (List[Dict[str, Any]], optional): Raw retrieved chunks for local fallback processing.

    Returns:
    str: Generated answer or standard fallback message.
    """
    if not chunks:
        return NOT_AVAILABLE_RESPONSE

    # Guardrail: If the most relevant chunk has a similarity score below SIMILARITY_THRESHOLD,
    # we determine the information is not present in the uploaded documents.
    max_similarity = max(c.get("similarity", 0.0) for c in chunks)
    if max_similarity < SIMILARITY_THRESHOLD:
        logger.info(f"Retrieved context maximum similarity is {max_similarity:.4f}, which is below the threshold of {SIMILARITY_THRESHOLD}. Returning standard fallback response.")
        return NOT_AVAILABLE_RESPONSE


    # Validate parameters
    temp = max(0.0, min(float(temperature), 1.0))
    p_val = max(0.0, min(float(top_p), 1.0))
    k_val = max(1, min(int(top_k), 100))

    try:
        logger.info(f"Generating LLM answer locally via Ollama ({OLLAMA_MODEL_NAME}) (Temp: {temp}, Top-P: {p_val}, Top-K: {k_val})...")
        response = _call_local_ollama(prompt_payload, temp, p_val, k_val)
        
        # Post-process check: if response states information is not available, return standard fallback
        resp_lower = response.lower()
        refusal_keywords = [
            "not mentioned", "no mention", "does not mention", 
            "not present in the context", "not found in the context",
            "not provided in the context", "does not contain", 
            "does not state", "no information", "cannot answer",
            "unable to answer", "context does not provide",
            "not available in the uploaded", "not available in the document",
            "does not explicitly mention", "does not explicitly state",
            "don't have relevant answer", "do not have relevant",
            "no relevant document"
        ]
        if any(rk in resp_lower for rk in refusal_keywords):
            logger.info(f"LLM response indicates missing info/refusal. Overriding with fallback response.")
            return NOT_AVAILABLE_RESPONSE

        if _is_response_grounded(response, chunks, prompt_payload.get("question", "")):
            return response
        else:
            logger.warning("Generated LLM response failed grounding validation. Overriding with fallback response.")
            return NOT_AVAILABLE_RESPONSE
    except Exception as e:
        logger.warning(f"Ollama local generation failed ({e}). Reverting to local extractive RAG engine...")

    # Fallback: Local Extractive RAG Engine if Ollama call fails
    return _local_extractive_generation(prompt_payload, chunks, temp, p_val, k_val)


def _is_response_grounded(response: str, chunks: List[Dict[str, Any]], question: str = "") -> bool:
    """
    Checks if the generated response is grounded in the retrieved chunks by verifying
    that the descriptive terms in the response actually exist within the chunk texts.
    """
    if not response:
        return False

    resp_lower = response.strip().lower()
    fallback_lower = NOT_AVAILABLE_RESPONSE.strip().lower()
    
    if (
        fallback_lower in resp_lower or 
        "don't have relevant answer" in resp_lower or 
        "no relevant document" in resp_lower or 
        "do not have relevant" in resp_lower
    ):
        return True

    resp_words = set(re.findall(r'\b\w{3,}\b', resp_lower))
    
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "you", "are", "not", "but", 
        "was", "were", "been", "have", "has", "had", "can", "could", "should", "would",
        "who", "what", "where", "when", "why", "how", "all", "any", "both", "each", "few",
        "more", "most", "other", "some", "such", "than", "too", "very", "she", "her", "him", 
        "his", "them", "their", "theirs", "its", "our", "ours", "your", "yours", "source",
        "page", "document", "answer", "question", "text", "context", "information", "according",
        "uploaded", "provided", "snippet", "file", "citation",
        "built", "build", "program", "programming", "language", "tracker", "project", 
        "support", "use", "using", "run", "running", "make", "made", "create", "created", 
        "develop", "developed", "write", "written", "code", "coded", "base", "based", 
        "features", "feature"
    }
    
    resp_keywords = resp_words - stop_words
    if not resp_keywords:
        return True

    combined_chunks_text = " ".join([c.get("text", "") for c in chunks]).lower()
    chunk_words = set(re.findall(r'\b\w{3,}\b', combined_chunks_text))
    chunk_word_roots = {get_word_root(cw) for cw in chunk_words}

    # Identify query terms that are completely absent from the context (excluding stop words)
    if question:
        q_words_clean = set(re.findall(r'\b\w{3,}\b', question.lower())) - stop_words
        out_of_context_terms = set()
        for qw in q_words_clean:
            qw_root = get_word_root(qw)
            if qw_root not in chunk_word_roots and not any(qw_root in cw or cw.startswith(qw_root) for cw in chunk_words):
                out_of_context_terms.add(qw)

        if out_of_context_terms:
            contained_terms = set()
            for rk in resp_keywords:
                rk_root = get_word_root(rk)
                for oct in out_of_context_terms:
                    oct_root = get_word_root(oct)
                    if rk_root == oct_root or rk.startswith(oct_root) or oct.startswith(rk_root):
                        contained_terms.add(rk)
            
            if contained_terms:
                # Check for negation in response
                negation_words = {"not", "no", "unable", "cannot", "future", "won't", "don't", "lack", "missing", "neither"}
                if not any(nw in resp_lower for nw in negation_words):
                    logger.warning(f"Grounding check: Response contains out-of-context terms {contained_terms} without negation. Flagging as ungrounded.")
                    return False

    matching_keywords = set()
    for rk in resp_keywords:
        rk_root = get_word_root(rk)
        if rk_root in chunk_word_roots or any(rk_root in cw or cw.startswith(rk_root) for cw in chunk_words):
            matching_keywords.add(rk)

    if not matching_keywords:
        logger.info("Grounding check: Zero matching keywords found between LLM response and retrieved context.")
        return False

    overlap_ratio = len(matching_keywords) / len(resp_keywords)
    logger.info(f"Grounding check: {len(matching_keywords)}/{len(resp_keywords)} keywords matched context (Ratio: {overlap_ratio:.4f})")

    if overlap_ratio < 0.25:
        return False

    return True



def _call_local_ollama(
    prompt_payload: Dict[str, str],
    temperature: float,
    top_p: float,
    top_k: int
) -> str:
    """
    Purpose:
    Calls the local Ollama instance using the official Ollama Python SDK.

    Parameters:
    prompt_payload (Dict[str, str]): Payload containing instruction sets.
    temperature (float): Generation temperature.
    top_p (float): Nucleus sampling rate.
    top_k (int): Top-K candidates.

    Returns:
    str: Text response from local Llama model.

    Raises:
    RuntimeError: If Ollama fails or returns invalid text.
    """
    system_instruction = prompt_payload.get("system_instruction", "")
    user_content = prompt_payload.get("user_content", "")

    try:
        import ollama
        client = ollama.Client(host=OLLAMA_HOST)
        
        response = client.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            options={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
        )

        if response and "message" in response and "content" in response["message"]:
            return response["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Ollama error: {str(e)}")

    raise RuntimeError("No valid response returned from local Ollama model.")


def _local_extractive_generation(
    prompt_payload: Dict[str, str],
    chunks: List[Dict[str, Any]],
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K
) -> str:
    """
    Purpose:
    Local Extractive Engine fallback. Synthesizes a direct, concise answer
    extracting target sentence(s) that directly address the user question,
    preventing full-document paragraph dumps.

    Parameters:
    prompt_payload (Dict[str, str]): Prompts containing query definitions.
    chunks (List[Dict[str, Any]]): Retrieved vector database documents.
    temperature (float): Randomness slider value (affects sorting perturbation).
    top_p (float): Nucleus probability threshold for sentence candidates.
    top_k (int): Limits maximum output sentences returned.

    Returns:
    str: Extracted concise text response.
    """
    if not chunks:
        return NOT_AVAILABLE_RESPONSE

    question = prompt_payload.get("question", "").strip()
    if not question:
        user_content = prompt_payload.get("user_content", "")
        question_match = re.search(r"USER QUESTION:\s*\n(.*?)\n\n(?:RETRIEVED DOCUMENT CONTEXT|REMINDER)", user_content, re.DOTALL)
        question = question_match.group(1).strip() if question_match else ""

    if not question:
        return NOT_AVAILABLE_RESPONSE

    # Comprehensive stop words set
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

    # Extract non-stopword query tokens
    query_tokens = [w.lower() for w in re.findall(r'\b\w+\b', question) if w.lower() not in stop_words and len(w) > 1]
    is_overview_q = any(w in question.lower() for w in ["summary", "summarize", "overview", "describe", "main point", "what is this document", "what is given in"])

    scored_sentences = []
    seen_text = set()

    for chunk in chunks:
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Document")
        page = meta.get("page", 1)

        # Split text into individual sentences / logical clauses
        raw_units = re.split(r'(?<=[.!?])\s+|\n+|(?:;\s+)', text)
        sentences = [u.strip() for u in raw_units if len(u.strip()) > 8]

        for sentence in sentences:
            # Truncate long unparsed blocks to first sentence or max 250 chars
            if len(sentence) > 250:
                sub_parts = re.split(r'(?<=[.!?])\s+', sentence)
                sentence_clean = sub_parts[0] if (sub_parts and len(sub_parts[0]) > 8) else sentence[:250].strip() + "..."
            else:
                sentence_clean = sentence.strip()

            norm_key = sentence_clean.lower()
            if norm_key in seen_text:
                continue
            seen_text.add(norm_key)

            sentence_words = set(w.lower() for w in re.findall(r'\b\w+\b', sentence_clean))
            
            # Score matching query tokens
            matched_count = 0.0
            for qt in query_tokens:
                if qt in sentence_words:
                    matched_count += 2.0
                elif any(qt in sw for sw in sentence_words):
                    matched_count += 1.0

            # Extra weight for exact multi-word phrase matches
            if len(query_tokens) >= 2:
                phrase = " ".join(query_tokens)
                if phrase in sentence_clean.lower():
                    matched_count += 4.0

            if matched_count > 0 or is_overview_q:
                scored_sentences.append({
                    "source": source,
                    "page": page,
                    "sentence": sentence_clean,
                    "score": matched_count if not is_overview_q else (matched_count if matched_count > 0 else 1.0)
                })

    if not scored_sentences:
        return NOT_AVAILABLE_RESPONSE

    # Apply Temperature logic: add random perturbation to the candidate score if temperature > 0
    if temperature > 0.0:
        import random
        for item in scored_sentences:
            item["score"] += random.uniform(-temperature * 2.0, temperature * 2.0)

    # Sort candidates by relevance score descending
    scored_sentences.sort(key=lambda x: x["score"], reverse=True)

    # Apply Top-P (nucleus filtering) to candidate sentences
    if top_p < 1.0 and len(scored_sentences) > 1:
        total_score = sum(max(0.0, s["score"]) for s in scored_sentences)
        if total_score > 0.0:
            cumulative_p = 0.0
            filtered = []
            for s in scored_sentences:
                prob = max(0.0, s["score"]) / total_score
                filtered.append(s)
                cumulative_p += prob
                if cumulative_p >= top_p:
                    break
            scored_sentences = filtered

    # Strict guardrail: If top score is <= 0 and not an overview query, return Not Available
    if scored_sentences[0]["score"] <= 0.0 and not is_overview_q:
        return NOT_AVAILABLE_RESPONSE

    # Limit output sentences dynamic bounds using top_k
    max_sentences = max(1, min(top_k, 5))
    top_matches = scored_sentences[:max_sentences]
    extracted_answer = " ".join([m["sentence"] for m in top_matches])

    # Deduplicate sources
    sources_list = []
    for item in top_matches:
        src_label = f"{item['source']} (Page {item['page']})"
        if src_label not in sources_list:
            sources_list.append(src_label)
            
    sources_str = ", ".join(sources_list)

    return f"{extracted_answer}\n\n*Source: {sources_str}*"


