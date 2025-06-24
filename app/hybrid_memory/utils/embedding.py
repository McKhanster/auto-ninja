# app/hybrid_memory/utils/embedding.py
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

def generate_embedding(text: str) -> list[float]:
    """Generate an embedding for a given text using a SentenceTransformer model."""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # Force CPU
        logger.info(f"Generating embedding for text: {text[:50]}...")
        embedding = model.encode([text])[0].tolist()
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise