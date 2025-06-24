# app/models/payload.py
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from ..config.settings import settings
import json

class GrokPayload(BaseModel):
    model: str = settings.GROK_MODEL
    messages: List[Dict[str, str]]
    max_tokens: int = settings.MAX_NEW_TOKENS
    temperature: float = 0.7
    top_p: float = 0.95

def create_grok_payload(
    prompt: str,
    system_message: Optional[str] = None,
    role: str = "user",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None
) -> GrokPayload:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": role, "content": prompt})
    payload = GrokPayload(messages=messages)
    if max_tokens is not None:
        payload.max_tokens = max_tokens
    if temperature is not None:
        payload.temperature = temperature
    if top_p is not None:
        payload.top_p = top_p
    return payload

def parse_grok_response(response: dict, expected_format: str = "text", validate_json: Optional[Dict[str, type]] = None) -> Union[str, dict]:
    if "choices" not in response or not response["choices"] or "message" not in response["choices"][0]:
        raise ValueError("Invalid Grok response format")
    content = response["choices"][0]["message"]["content"]
    if expected_format == "text":
        return content
    if expected_format == "json":
        try:
            parsed = json.loads(content)
            if validate_json:
                for key, expected_type in validate_json.items():
                    if key not in parsed or not isinstance(parsed[key], expected_type):
                        raise ValueError(f"Invalid JSON structure for key '{key}'")
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Grok response as JSON: {str(e)}")
    raise ValueError(f"Unsupported expected_format: {expected_format}")