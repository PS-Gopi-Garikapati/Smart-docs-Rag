# API Documentation

The **Smart Document Assistant** API exposes endpoints for document lifecycle management and semantic query execution.

## Endpoints

### 1. Healthcheck
- **Endpoint**: `/api/health`
- **Method**: `GET`
- **Response Model**: JSON
- **Success Response (200 OK)**:
```json
{
  "status": "online",
  "service": "Smart Document Assistant RAG API",
  "version": "1.0.0"
}
```

### 2. Document Upload
- **Endpoint**: `/api/upload`
- **Method**: `POST`
- **Payload**: `multipart/form-data` with `files` list parameter.
- **Success Response (201 Created)**:
```json
{
  "status": "success",
  "message": "Successfully processed 1 document(s).",
  "total_chunks_indexed": 12,
  "documents": [
    {
      "filename": "sample.pdf",
      "pages_parsed": 2,
      "chunks_indexed": 12
    }
  ]
}
```

### 3. List Indexed Documents
- **Endpoint**: `/api/documents`
- **Method**: `GET`
- **Success Response (200 OK)**:
```json
{
  "status": "success",
  "count": 1,
  "documents": [
    {
      "filename": "sample.pdf",
      "chunk_count": 12
    }
  ]
}
```

### 4. Delete Single Document
- **Endpoint**: `/api/documents/{filename}`
- **Method**: `DELETE`
- **Success Response (200 OK)**:
```json
{
  "status": "success",
  "message": "Document 'sample.pdf' and 12 vector chunks removed successfully.",
  "deleted_chunks": 12
}
```

### 5. Clear All Documents
- **Endpoint**: `/api/clear`
- **Method**: `DELETE`
- **Success Response (200 OK)**:
```json
{
  "status": "success",
  "message": "All uploaded documents and vector index have been cleared successfully."
}
```

### 6. RAG Semantic Query
- **Endpoint**: `/api/query`
- **Method**: `POST`
- **Request Body (JSON)**:
```json
{
  "question": "What is the company profit margin?",
  "temperature": 0.3,
  "top_p": 0.9,
  "top_k": 3
}
```
- **Success Response (200 OK)**:
```json
{
  "status": "success",
  "question": "What is the company profit margin?",
  "answer": "The company profit margin is 18.5%. *Source: sample.pdf (Page 1)*",
  "retrieved_sources": [
    {
      "citation_id": 1,
      "source": "sample.pdf",
      "page": 1,
      "similarity_score": 0.8921,
      "snippet": "Our quarterly reports demonstrate that the company profit margin has stabilized at 18.5%..."
    }
  ],
  "execution_time_seconds": 0.145,
  "parameters_used": {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 3
  }
}
```
