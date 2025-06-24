from ..hybrid_memory.core import HybridMemory
from ..models.interaction import Interaction
from ..models.agent import Agent
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, sqlite_path: str, qdrant_path: str):
        self.memory = HybridMemory(
            sqlite_path=sqlite_path,
            qdrant_path=qdrant_path,
            embedding_model="all-MiniLM-L6-v2"
        )
        self.memory.initialize()
        self.current_agent = None
        self.intent_recognizer = None

    def set_current_agent(self, agent: Agent):
        self.current_agent = agent
        logger.info(f"Current agent set to: {agent.name} (ID: {agent.id})")

    def set_intent_recognizer(self, intent_recognizer):
        self.intent_recognizer = intent_recognizer
        logger.info("Intent recognizer set for MemoryManager")

    def process(self, user_prompt: str, local_output: str, final_output: str) -> str:
        if not self.current_agent:
            raise ValueError("No current agent set. Please create or select an agent first.")

        intent, params = self.intent_recognizer.recognize_intent(user_prompt) if self.intent_recognizer else ("other", {})
        interaction = Interaction(
            agent_id=self.current_agent.id,
            user_prompt=user_prompt,
            actual_output=local_output,
            target_output=final_output,
            intent=intent,
            parent_id=None,
            metadata={"parameters": params, "user_id": "user123", "session_id": "session456"}
        )
        self.memory.store_interaction(interaction)
        logger.info(f"Stored interaction with intent: {intent}")
        return final_output

    def get_interaction(self, interaction_id: int) -> Interaction:
        return self.memory.get_interaction(interaction_id)

    def get_similar_interactions(self, prompt: str, limit: int = 3) -> list[Interaction]:
        return self.memory.get_similar_interactions(prompt, limit=limit)

    def get_context(self) -> str:
        if not self.current_agent:
            return ""
        context = f"Agent Role: {self.current_agent.role}\n"
        if self.current_agent.role_info:
            context += f"Knowledge: {self.current_agent.role_info.get('knowledge', '')}\n"
            skills = [skill["name"] if isinstance(skill, dict) else str(skill) for skill in self.current_agent.role_info.get('skills', [])]
            context += f"Skills: {', '.join(skills)}\n"
            context += f"Tools: {', '.join(self.current_agent.role_info.get('tools', []))}\n"
        return context

    def learn(self):
        print("Learning not yet implemented in this minimal version.")