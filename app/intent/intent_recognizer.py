import sqlite3
import json
import numpy as np
import logging
from ..shared import SingletonSentenceTransformer

logger = logging.getLogger(__name__)

class IntentRecognizer:
    def __init__(self, inference_engine, db_path: str):
        self.inference_engine = inference_engine
        self.db_path = db_path
        self.embedding_model = SingletonSentenceTransformer.get_instance()
        logger.info("Initialized IntentRecognizer with shared SentenceTransformer")

    def recognize_intent(self, prompt: str) -> tuple[str, dict]:
        """Classify prompt into an intent and extract parameters."""
        try:
            # Step 1: Embedding-based recognition
            embedding = self.embedding_model.encode([prompt])[0]
            logger.debug(f"Input embedding shape for prompt '{prompt}': {embedding.shape}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT intent, embedding, parameters FROM action_intents")
            best_intent, best_score = "other", 0.0
            best_params = {}
            for intent, emb_bytes, params in cursor.fetchall():
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                logger.debug(f"Stored embedding for intent '{intent}': bytes {len(emb_bytes)}, shape {emb.shape}")
                score = np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb))
                if score > best_score and score > 0.8:
                    best_intent, best_score = intent, score
                    best_params = json.loads(params) if params else {}
            conn.close()
            logger.info(f"Embedding intent: {best_intent}, score: {best_score}")

            # Step 2: Grok backup if online
            if best_intent == "other" and hasattr(self.inference_engine, 'grok_enabled') and self.inference_engine.grok_enabled:
                try:
                    grok_response = self.inference_engine._predict_grok(
                        prompt,
                        "Classify intent: exit, use_speech, use_text, list_skills, acquire_skill, use_skill, "
                        "update_skill, delete_skill, acquire_next, script_generate, script_execute, script_update, "
                        "script_delete, other. Return JSON: {'intent': '...', 'parameters': {...}}. "
                        "Extract parameters like skill_id, task, or file_name if applicable."
                    )
                    grok_data = json.loads(grok_response)
                    best_intent = grok_data["intent"]
                    best_params = grok_data.get("parameters", {})
                    logger.info(f"Grok intent: {best_intent}")
                except Exception as e:
                    logger.error(f"Grok intent failed: {e}")

            # Step 3: Gemma offline fallback
            if best_intent == "other":
                try:
                    local_response = self.inference_engine._predict_local(
                        f"Input: {prompt}\n"
                        "Classify intent: exit, use_speech, use_text, list_skills, acquire_skill, use_skill, "
                        "update_skill, delete_skill, acquire_next, script_generate, script_execute, script_update, "
                        "script_delete, other\n"
                        "Return JSON: {{'intent': '...', 'parameters': {{}}}}\n"
                        "Extract parameters like skill_id, task, or file_name if applicable."
                    )
                    local_data = json.loads(local_response)
                    best_intent = local_data["intent"]
                    best_params = local_data.get("parameters", {})
                    logger.info(f"Gemma intent: {best_intent}")
                except Exception as e:
                    logger.error(f"Gemma intent failed: {e}")
                    best_intent = "other"
                    best_params = {}

            # Parameter parsing for specific intents
            if best_intent in ["acquire_skill", "use_skill", "update_skill", "delete_skill", "acquire_next", 
                             "script_execute", "script_update", "script_delete"] and not best_params:
                best_params = self._parse_parameters(prompt, best_intent)
            
            return best_intent, best_params
        except Exception as e:
            logger.error(f"Intent recognition failed: {e}")
            return "other", {}

    def _parse_parameters(self, prompt: str, intent: str) -> dict:
        """Parse parameters from prompt based on intent."""
        params = {}
        try:
            parts = prompt.split(" ", 2)
            if intent == "acquire_skill" and len(parts) > 2:
                name_desc = parts[2].split(",", 1)
                params["name"] = name_desc[0].strip()
                params["description"] = name_desc[1].strip() if len(name_desc) > 1 else ""
            elif intent == "use_skill" and len(parts) > 2:
                skill_id_task = parts[2].split(",", 1)
                params["skill_id"] = skill_id_task[0].strip()
                params["task"] = skill_id_task[1].strip() if len(skill_id_task) > 1 else ""
            elif intent == "update_skill" and len(parts) > 2:
                skill_id_desc = parts[2].split(",", 1)
                params["skill_id"] = skill_id_desc[0].strip()
                params["description"] = skill_id_desc[1].strip() if len(skill_id_desc) > 1 else ""
            elif intent == "delete_skill" and len(parts) > 2:
                params["skill_id"] = parts[2].strip()
            elif intent == "acquire_next" and len(parts) > 2:
                params["count"] = parts[2].strip()
            elif intent in ["script_execute", "script_update", "script_delete"] and len(parts) > 2:
                params["file_name"] = parts[2].strip()
        except Exception as e:
            logger.error(f"Parameter parsing failed: {e}")
        return params