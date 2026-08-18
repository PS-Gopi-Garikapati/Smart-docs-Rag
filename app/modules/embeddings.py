"""
Embeddings Module.

Manages vector embedding generation for document chunks and user queries.
Primary engine: SentenceTransformers (all-MiniLM-L6-v2).
Fallback engine: TF-IDF / Subword Dense Vector Hashing if PyTorch DLL errors occur on Windows.
"""

import logging
import subprocess
import sys
import hashlib
import re
from typing import List
import numpy as np
from app.config import EMBEDDING_MODEL_NAME

logger: logging.Logger = logging.getLogger(__name__)

# Global model and fallback state
_model_instance = None
_fallback_mode: bool = False
_torch_checked: bool = False
_torch_available: bool = False


def _check_pytorch_working() -> bool:
    """
    Purpose:
    Probes if PyTorch can be imported safely without triggering OS level DLL aborts on Windows.

    Returns:
    bool: True if PyTorch and sentence_transformers can be successfully imported, otherwise False.
    """
    global _torch_checked, _torch_available
    if _torch_checked:
        return _torch_available

    try:
        res = subprocess.run(
            [sys.executable, "-c", "import torch; from sentence_transformers import SentenceTransformer"],
            capture_output=True,
            timeout=5
        )
        _torch_available = (res.returncode == 0)
    except Exception:
        _torch_available = False

    _torch_checked = True
    return _torch_available


def _init_model() -> None:
    """
    Purpose:
    Initializes the embedding engine. Attempts SentenceTransformer if PyTorch is functional,
    otherwise uses a high-quality zero-dependency dense feature vectorizer fallback.
    """
    global _model_instance, _fallback_mode

    if _model_instance is not None or _fallback_mode:
        return

    if _check_pytorch_working():
        try:
            logger.info(f"Attempting to load SentenceTransformer model '{EMBEDDING_MODEL_NAME}'...")
            from sentence_transformers import SentenceTransformer
            _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
            _fallback_mode = False
            logger.info("SentenceTransformer embedding engine loaded successfully.")
            return
        except Exception as e:
            logger.warning(f"SentenceTransformer load failed ({e}). Switching to dense vector fallback engine.")

    logger.info("Operating in zero-dependency TF-IDF / Subword Dense Vector Fallback Mode.")
    _fallback_mode = True


def generate_text_embedding(text: str) -> List[float]:
    """
    Purpose:
    Generates a dense vector embedding representation for a single text string.

    Parameters:
    text (str): Input text string.

    Returns:
    List[float]: Vector embedding representing semantic content.

    Raises:
    ValueError: If input text string is empty.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text string.")

    _init_model()

    if not _fallback_mode and _model_instance is not None:
        try:
            vec = _model_instance.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return vec.tolist()
        except Exception as e:
            logger.warning(f"SentenceTransformer encoding error ({e}), using dense hash encoding.")

    # Fallback dense subword embedding
    return _generate_dense_fallback_embedding(text)


def generate_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Purpose:
    Generates vector embeddings for a list of text strings in batch mode.

    Parameters:
    texts (List[str]): List of text snippets to encode.

    Returns:
    List[List[float]]: List of vector embeddings matching input text count.
    """
    if not texts:
        return []

    _init_model()

    if not _fallback_mode and _model_instance is not None:
        try:
            vecs = _model_instance.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            return vecs.tolist()
        except Exception as e:
            logger.warning(f"SentenceTransformer batch encoding error ({e}), using dense hash encoding.")

    return [_generate_dense_fallback_embedding(t) for t in texts]


def _generate_dense_fallback_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Purpose:
    Generates a normalized 384-dimensional dense feature vector using character/subword n-grams
    and word hashing for zero-dependency semantic vector similarity when PyTorch DLLs are absent.

    Parameters:
    text (str): Input text string.
    dim (int): Vector dimensionality (defaults to 384).

    Returns:
    List[float]: Generated normalized dense feature vector.
    """
    clean_text = text.lower()
    words = [w for w in re.findall(r'\w+', clean_text) if len(w) > 1]
    
    stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "it", "this", "that"}
    
    vec = np.zeros(dim, dtype=np.float32)

    for idx, word in enumerate(words):
        # Base weight based on word length and stopword filtering
        weight = 0.2 if word in stop_words else 1.0
        pos_decay = 1.0 / (1.0 + 0.05 * idx)

        # Full word hash
        hw = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
        vec[hw % dim] += 2.0 * weight * pos_decay

        # Character 3-gram and 4-gram hashing
        if len(word) >= 3:
            for n in (3, 4):
                for i in range(len(word) - n + 1):
                    ngram = word[i:i+n]
                    hn = int(hashlib.md5(ngram.encode('utf-8')).hexdigest(), 16)
                    vec[hn % dim] += 0.5 * weight * pos_decay

        # Word bigram hashing
        if idx < len(words) - 1:
            next_word = words[idx + 1]
            bigram = f"{word}_{next_word}"
            hb = int(hashlib.md5(bigram.encode('utf-8')).hexdigest(), 16)
            vec[hb % dim] += 1.2 * weight * pos_decay

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


