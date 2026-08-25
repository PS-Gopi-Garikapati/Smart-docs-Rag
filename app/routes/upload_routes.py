"""
Upload Routes Module.

FastAPI endpoint routes for multi-format document upload (PDF, DOCX, CSV, TXT, MD, JSON),
text extraction, document chunking, vector embedding indexing, document listing, and state clearing.
"""

import os
import shutil
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from app.config import UPLOAD_DIR
from app.modules.document_processor import extract_text_from_file, chunk_document_pages
from app.modules.embeddings import generate_batch_embeddings
from app.modules.vector_store import add_chunks_to_vector_store, get_indexed_documents, clear_vector_store, delete_document_from_vector_store

logger: logging.Logger = logging.getLogger(__name__)

router: APIRouter = APIRouter(prefix="/api", tags=["Document Management"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".csv", ".txt", ".md", ".json", ".log", ".text"}


class DocumentSummary(BaseModel):
    filename: str
    pages_parsed: int
    chunks_indexed: int


class UploadResponse(BaseModel):
    status: str
    message: str
    total_chunks_indexed: int
    documents: List[DocumentSummary]


class ListDocumentsResponse(BaseModel):
    status: str
    count: int
    documents: List[Dict[str, Any]]


class DeleteResponse(BaseModel):
    status: str
    message: str
    deleted_chunks: int


class ClearResponse(BaseModel):
    status: str
    message: str


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(files: List[UploadFile] = File(...)) -> UploadResponse:
    """
    Purpose:
    Handles multi-format document file uploads (PDF, Word, Text, Markdown).

    Processing Steps:
    1. Validates supported document file extension.
    2. Saves file to local `data/uploads/` storage directory.
    3. Cleans up any existing chunks for re-uploaded files.
    4. Extracts text according to file format.
    5. Splits text into overlapping chunks.
    6. Generates vector embeddings using SentenceTransformers.
    7. Stores vectors and metadata in ChromaDB.

    Parameters:
    files (List[UploadFile]): List of uploaded file streams.

    Returns:
    UploadResponse: Upload metrics and processed summaries.

    Raises:
    HTTPException: 400 Bad Request on invalid extensions, 500 on server processing issues.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were provided in the upload request."
        )

    processed_summary = []
    total_chunks_added = 0

    for file in files:
        if not file.filename:
            continue
        
        # Enforce upload size limit (200MB = 200 * 1024 * 1024 bytes)
        MAX_SIZE = 200 * 1024 * 1024
        try:
            file.file.seek(0, 2)
            file_size = file.file.tell()
            file.file.seek(0)
        except Exception as e:
            logger.warning(f"Could not determine file size for '{file.filename}': {e}")
            file_size = 0

        if file_size > MAX_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds the 200MB limit for '{file.filename}'."
            )

        # Sanitize filename to prevent directory traversal and invalid characters
        import re
        original_name = file.filename
        base_name, ext = os.path.splitext(original_name)
        sanitized_base = re.sub(r'[^a-zA-Z0-9_\-]', '_', base_name)
        sanitized_filename = f"{sanitized_base}{ext.lower()}"

        # Validate file extension
        if ext.lower() not in ALLOWED_EXTENSIONS:
            allowed_list = ", ".join(sorted(list(ALLOWED_EXTENSIONS)))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type for '{original_name}'. Supported formats are: {allowed_list}"
            )

        saved_path = os.path.join(UPLOAD_DIR, sanitized_filename)

        try:
            # Clean up existing vector store chunks if re-uploading file with same name
            delete_document_from_vector_store(sanitized_filename)

            # Save uploaded file stream to storage
            with open(saved_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            logger.info(f"Saved uploaded file: '{sanitized_filename}' to {saved_path}")

            # Step 1: Extract text per page/section based on format
            pages_content = extract_text_from_file(saved_path)

            # Step 2: Chunk pages into semantic snippets
            chunks = chunk_document_pages(pages_content, doc_name=sanitized_filename)

            # Step 3: Extract texts and generate batch embeddings
            chunk_texts = [c["text"] for c in chunks]
            embeddings = generate_batch_embeddings(chunk_texts)

            # Step 4: Add chunks and vectors to ChromaDB store
            added_count = add_chunks_to_vector_store(chunks, embeddings)
            total_chunks_added += added_count

            processed_summary.append(DocumentSummary(
                filename=sanitized_filename,
                pages_parsed=len(pages_content),
                chunks_indexed=added_count
            ))

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing upload for file '{sanitized_filename}': {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process '{sanitized_filename}': {str(e)}"
            )

    return UploadResponse(
        status="success",
        message=f"Successfully processed {len(files)} document(s).",
        total_chunks_indexed=total_chunks_added,
        documents=processed_summary
    )



@router.get("/documents", response_model=ListDocumentsResponse)
async def list_documents() -> ListDocumentsResponse:
    """
    Purpose:
    Retrieves summary list of all currently indexed documents in ChromaDB vector store.

    Returns:
    ListDocumentsResponse: File listings and counts.
    """
    try:
        from app.modules.vector_store import sync_vector_store_with_uploads
        sync_vector_store_with_uploads()
        documents = get_indexed_documents()
        return ListDocumentsResponse(
            status="success",
            count=len(documents),
            documents=documents
        )
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document inventory: {str(e)}"
        )


@router.delete("/documents/{filename}", response_model=DeleteResponse)
async def delete_single_document(filename: str) -> DeleteResponse:
    """
    Purpose:
    Deletes a specific uploaded PDF document file and its vector embeddings from ChromaDB.

    Parameters:
    filename (str): Name of target document file to remove.

    Returns:
    DeleteResponse: Removal status and confirmation parameters.
    """
    try:
        # Delete chunks from ChromaDB
        deleted_count = delete_document_from_vector_store(filename)

        # Delete physical file from uploads folder
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        return DeleteResponse(
            status="success",
            message=f"Document '{filename}' and {deleted_count} vector chunks removed successfully.",
            deleted_chunks=deleted_count
        )
    except Exception as e:
        logger.error(f"Error deleting document '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document '{filename}': {str(e)}"
        )


@router.delete("/clear", response_model=ClearResponse)
async def clear_all_documents() -> ClearResponse:
    """
    Purpose:
    Clears all vector embeddings from ChromaDB and deletes uploaded PDF files.

    Returns:
    ClearResponse: Success notification.
    """
    try:
        # Clear ChromaDB vector database
        clear_vector_store()

        # Delete files from upload directory
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        return ClearResponse(
            status="success",
            message="All uploaded documents and vector index have been cleared successfully."
        )
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear vector store: {str(e)}"
        )


