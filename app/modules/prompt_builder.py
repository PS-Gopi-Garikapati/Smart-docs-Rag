"""
Prompt Builder Module.
Constructs strict system prompts and context payload structures to ensure the LLM
answers exclusively based on retrieved document content.
"""

from typing import List, Dict, Any
from app.config import NOT_AVAILABLE_RESPONSE


SYSTEM_PROMPT = """You are a highly precise Smart Document Assistant powered by Retrieval-Augmented Generation (RAG).

CRITICAL RESPONSE RULES:
1. Answer the user's question ONLY and EXCLUSIVELY using the retrieved document context below.
2. Do NOT use any pre-trained general knowledge, external information, or assumptions to answer the question.
3. If the answer to the user's question cannot be found or directly inferred from the retrieved document context below, you MUST respond ONLY with the exact phrase:
   "{not_available_msg}"
4. Do NOT output any other text, warnings, prefaces, or explanations. If the answer is not in the context, your entire response must be exactly "{not_available_msg}".
5. End your answer with a brief citation format, e.g. "*Source: document_name.pdf (Page X)*".
""".format(not_available_msg=NOT_AVAILABLE_RESPONSE)



def format_context_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Purpose:
    Formats a list of retrieved chunk dictionaries into a readable context block for the LLM prompt.

    Parameters:
    chunks (List[Dict[str, Any]]): Retrieved text chunks with metadata.

    Returns:
    str: Formatted context string with document and page citations.
    """
    if not chunks:
        return "No relevant document context found."

    context_blocks = []
    for idx, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source_name = meta.get("source", "Unknown Document")
        page_num = meta.get("page", "?")
        text = chunk.get("text", "").strip()

        block = f"--- CONTEXT SNIPPET #{idx} [Source: {source_name} | Page: {page_num}] ---\n{text}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_rag_prompt(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Purpose:
    Assembles system prompt and user prompt payload for sending to the LLM.

    Parameters:
    question (str): User question text.
    chunks (List[Dict[str, Any]]): List of retrieved relevant document chunks.

    Returns:
    Dict[str, str]: Dictionary containing 'system_instruction' and 'user_content'.
    """
    formatted_context = format_context_from_chunks(chunks)

    user_content = f"""USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
=========================================
{formatted_context}
=========================================

REMINDER: Answer ONLY the question directly in 1-3 sentences. Do not dump or copy the context text. If the answer is not present in the context above, respond ONLY with:
"{NOT_AVAILABLE_RESPONSE}"
"""

    return {
        "system_instruction": SYSTEM_PROMPT,
        "user_content": user_content,
        "question": question
    }



