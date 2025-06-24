# app/speech/tts.py
from gtts import gTTS
import pyttsx3
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(self, engine: str = settings.TTS_ENGINE):
        """Initialize TTS engine (gTTS or pyttsx3)."""
        self.engine = engine
        logger.info(f"Initializing TTS with engine: {self.engine}")
        if self.engine == "gTTS":
            self.tts = None  # gTTS instantiated per synthesize call
        elif self.engine == "pyttsx3":
            self.tts = pyttsx3.init()
            self.tts.setProperty("rate", 150)
            self.tts.setProperty("volume", 1.0)
        else:
            logger.error(f"Unsupported TTS engine: {self.engine}, falling back to pyttsx3")
            self.engine = "pyttsx3"
            self.tts = pyttsx3.init()
            self.tts.setProperty("rate", 150)
            self.tts.setProperty("volume", 1.0)

    def synthesize(self, text: str, output_path: str) -> None:
        """Convert text to speech and save to output_path."""
        try:
            logger.info(f"Synthesizing text with {self.engine}: {text[:50]}...")
            if self.engine == "gTTS":
                tts = gTTS(text=text, lang="en", slow=False)
                tts.save(output_path)
            elif self.engine == "pyttsx3":
                self.tts.save_to_file(text, output_path)
                self.tts.runAndWait()
            logger.info(f"Audio saved to {output_path}")
        except Exception as e:
            logger.error(f"TTS synthesis failed with {self.engine}: {str(e)}")
            raise ValueError(f"Failed to synthesize audio: {str(e)}")