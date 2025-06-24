from ..config.settings import settings

class DataTransformer:
    def preprocess(self, message: str) -> str:
        # Basic validation
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message must be a non-empty string")
        return message.strip()
    
    def postprocess(self, text: str) -> str:
        cleaned_text = " ".join(text.split())
        return cleaned_text[:settings.MAX_SEQUENCE_LENGTH * 2]