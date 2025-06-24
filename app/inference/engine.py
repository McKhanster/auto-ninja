# app/inference/engine.py
import logging

import re
from app.models.payload import create_grok_payload, parse_grok_response
from ..transformers.data import DataTransformer
from ..config.settings import settings
from ..middleware.memory import MemoryManager
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import json

logger = logging.getLogger(__name__)

class AutoNinjaInference:
    def __init__(self):
        self.transformer = DataTransformer()
        self.device = torch.device(settings.DEVICE_TYPE if torch.cuda.is_available() and settings.DEVICE_TYPE != "cpu" else "cpu")
        self.model_name = settings.MODEL_NAME
        
        kwargs = {"token": settings.HF_TOKEN} if settings.HF_TOKEN else {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.memory_manager = MemoryManager(
            sqlite_path="./memory.db",
            qdrant_path="./qdrant_data"
        )
        
        self.grok_enabled = self.check_connectivity()
        self.grok_url = settings.GROK_URL
        self.grok_headers = {
            "Authorization": f"Bearer {settings.XAI_API_KEY}",
            "Content-Type": "application/json"
        } if self.grok_enabled else None
    
    def process(self, message: str) -> str:  # Removed script_manager parameter
        input_text = self._format_input(message)
        local_output = self._predict_local(input_text)
        if self.grok_enabled:
            try:
                grok_output = self._predict_grok(message, local_output)
                final_output = self.memory_manager.process(message, local_output, grok_output)
            except (requests.RequestException, ValueError) as e:
                logger.error(f"Grok API error: {str(e)}")
                final_output = self.memory_manager.process(message, local_output, local_output)
        else:
            final_output = self.memory_manager.process(message, local_output, local_output)
        return final_output
    
    def get_role_info(self, role: str) -> dict:
        if not self.grok_enabled:
            logger.warning("Grok disabled; returning default role info")
            return {"knowledge": f"Basic {role} knowledge", "skills": [f"{role} Basics"], "tools": []}
        try:
            system_message = "You are an expert in role assumption for AI agents."
            prompt = f"""
                Based on the role '{role}', provide detailed information in a structured format:
                - knowledge: A brief description of the knowledge required for the role.
                - skills: A list of skills with descriptions and, where applicable, Python code for automation.
                - tools: A list of tools useful for the role.

                Return the response in the following JSON format:
                {{
                  "knowledge": "...",
                  "skills": [
                    {{"name": "skill1", "description": "description1", "code": "python code or null"}},
                    ...
                  ],
                  "tools": ["tool1", "tool2", ...]
                }}
                """
            payload = create_grok_payload(
                prompt=prompt,
                system_message=system_message,
                max_tokens=4096,  # Increased for complete responses
                temperature=0.5
            )
            response = requests.post(self.grok_url, json=payload.model_dump(), headers=self.grok_headers)
            response.raise_for_status()
            raw_response = response.json()
            logger.info(f"Raw Grok response for role '{role}': {raw_response}")
            
            if not raw_response or "choices" not in raw_response or not raw_response["choices"]:
                raise ValueError("Empty or invalid Grok response structure")
            
            content = raw_response["choices"][0]["message"]["content"]
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content
            
            # Handle potentially truncated JSON
            try:
                role_info = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed, attempting to fix truncated response: {e}")
                # Attempt to close incomplete structures
                json_str = json_str.rstrip() + ']}' if json_str.endswith('[') or json_str.endswith(',') else json_str
                if not json_str.endswith('}'):
                    json_str += '}'
                role_info = json.loads(json_str)
            
            if not isinstance(role_info, dict) or not all(k in role_info for k in ["knowledge", "skills", "tools"]):
                raise ValueError("Grok response lacks required fields")
            if not isinstance(role_info["skills"], list) or not all(isinstance(s, dict) and "name" in s for s in role_info["skills"]):
                raise ValueError("Grok response 'skills' must be a list of dictionaries")
            if not isinstance(role_info["tools"], list) or not all(isinstance(t, str) for t in role_info["tools"]):
                raise ValueError("Grok response 'tools' must be a list of strings")
            return role_info
        except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get role info from Grok: {str(e)}")
            return {
                "knowledge": f"Basic understanding of {role} principles",
                "skills": [{"name": f"{role} Fundamentals", "description": "Basic tasks", "code": None}],
                "tools": ["Generic Toolset"]
            }
        

    def _format_input(self, message: str) -> str:
        return f"<|user|> {message} <|assistant|> "

    def _predict_local(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=settings.MAX_SEQUENCE_LENGTH,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_start = generated_text.find("<|assistant|> ") + len("<|assistant|> ")
        return generated_text[assistant_start:].strip() if assistant_start > len("<|assistant|> ") - 1 else generated_text.strip()

    def _predict_grok(self, user_message: str, local_output: str) -> str:
        context = self.memory_manager.get_context()
        prompt = f"{context}\nUser: {user_message}\nLocal Agent: {local_output}\nPlease provide a coherent, standalone response combining the user's input and the local agent's response, without repeating the prompt or labels."
        payload = create_grok_payload(
            prompt=prompt,
            max_tokens=4096,  # Increased for complete responses
            temperature=0.7  # Slightly higher for better code generation
        )
        response = requests.post(self.grok_url, json=payload.model_dump(), headers=self.grok_headers)
        response.raise_for_status()
        raw_response = response.json()
        logger.info(f"Raw Grok response in _predict_grok: {raw_response}")
        return parse_grok_response(raw_response, expected_format="text")

    def check_connectivity(self) -> bool:
        try:
            response = requests.get("https://google.com", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False