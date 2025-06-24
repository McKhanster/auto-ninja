import whisper
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)

class SpeechToText:
    def __init__(self, model_name: str = settings.WHISPER_MODEL):
        """Initialize Whisper model for STT."""
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        self.options = {"fp16": False}  # Use FP32 for broader compatibility

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text using Whisper."""
        try:
            logger.info(f"Transcribing audio from {audio_path}")
            result = self.model.transcribe(audio_path, **self.options)
            text = result["text"].strip()
            logger.info(f"Transcription result: {text}")
            return text
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise ValueError(f"Failed to transcribe audio: {str(e)}")