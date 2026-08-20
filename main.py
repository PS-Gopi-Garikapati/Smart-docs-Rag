"""
Main FastAPI Application Entrypoint.
Initializes FastAPI, configures CORS middleware, mounts static frontend files,
and registers document management and RAG query route endpoints.
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routes.upload_routes import router as upload_router
from app.routes.query_routes import router as query_router

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# Initialize FastAPI application
app = FastAPI(
    title="Smart Document Assistant API",
    description="Retrieval-Augmented Generation (RAG) backend powered by SentenceTransformers, ChromaDB, and Google Gemini API.",
    version="1.0.0"
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Route Modules
app.include_router(upload_router)
app.include_router(query_router)

# Mount Frontend Static Files Directory
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    async def serve_index():
        """
        Serves the single-page HTML frontend client.
        """
        return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/api/health")
async def health_check():
    """
    Application healthcheck endpoint.
    """
    return {
        "status": "online",
        "service": "Smart Document Assistant RAG API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT
    
    logger.info(f"Starting Smart Document Assistant API... on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True, reload_dirs=["app", "main.py"])


