import logging
from importlib.metadata import version
from fastapi import FastAPI
from .api.inference.endpoints import router as inference_router
from .shared import inference_engine  # Import singleton
from .middleware.memory import MemoryManager  # Explicit import for clarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting Auto Ninja with the following package versions:")
try:
    logger.info(f"torch: {version('torch')}")
    logger.info(f"transformers: {version('transformers')}")
    logger.info(f"torchvision: {version('torchvision')}")
    logger.info(f"sentence-transformers: {version('sentence-transformers')}")
    logger.info(f"numpy: {version('numpy')}")
    logger.info("Pre-loading inference engine model...")
    logger.info("Inference engine model pre-loaded")
    # Log speech-related dependencies
    logger.info(f"openai-whisper: {version('openai-whisper')}")
    logger.info(f"gtts: {version('gtts')}")
    logger.info(f"pyttsx3: {version('pyttsx3')}")
    logger.info(f"pyaudio: {version('pyaudio')}")
    logger.info(f"pydub: {version('pydub')}")
except Exception as e:
    logger.warning(f"Could not log version for some dependencies: {str(e)}")

app = FastAPI()
app.include_router(inference_router)