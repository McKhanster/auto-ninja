from pathlib import Path
from dotenv import load_dotenv
import os
import requests

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Settings:
    MODEL_NAME: str = os.getenv("MODEL_NAME", "google/gemma-3-1b-it")
    DEVICE_TYPE: str = os.getenv("DEVICE_TYPE", "cuda").lower()
    MAX_SEQUENCE_LENGTH: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "128"))
    HF_TOKEN: str = os.getenv("HF_TOKEN", None)
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "50"))
    XAI_API_KEY: str = os.getenv("XAI_API_KEY", None)
    GROK_MODEL: str = "grok-3-mini-beta"
    CHAT_COMPLETION_URL: str = "https://api.x.ai/v1/chat/completions"
    IMAGE_GENERATION_URL = str(os.getenv("IMAGE_GENERATION_URL", "https://api.x.ai/v1/image/generations"))
    GROK_URL = CHAT_COMPLETION_URL
    VALID_DEVICES = {"cpu", "gpu", "cuda"}
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "tiny")
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", None)
    SPEECH_DIR: str = os.getenv("SPEECH_DIR", "speech/")
    REQUEST_TIMEOUT: float = float(os.getenv("REQUEST_TIMEOUT", "30"))  # Default 30s

    def __init__(self):
        if self.DEVICE_TYPE not in self.VALID_DEVICES:
            raise ValueError(f"Invalid DEVICE_TYPE: {self.DEVICE_TYPE}. Must be one of {self.VALID_DEVICES}")
        if not self.XAI_API_KEY:
            print("Warning: XAI_API_KEY not set; Grok API integration will be disabled")
        if "llama" in self.MODEL_NAME.lower() and not self.HF_TOKEN:
            print("Warning: HF_TOKEN might be required for LLaMA models")
        Path(self.SPEECH_DIR).mkdir(exist_ok=True)
        if not self.TTS_ENGINE:
            self.TTS_ENGINE = "gTTS" if self.check_connectivity() and self.XAI_API_KEY else "pyttsx3"
            print(f"TTS_ENGINE set to {self.TTS_ENGINE}")

    def check_connectivity(self) -> bool:
        try:
            response = requests.get("https://google.com", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

settings = Settings()