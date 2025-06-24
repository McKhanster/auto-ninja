from datetime import datetime
import json

class Interaction:
    def __init__(self, id=None, agent_id=None, timestamp=None, user_prompt="", actual_output="", target_output="", intent=None, parent_id=None, metadata=None):
        self.id = id
        self.agent_id = agent_id
        self.timestamp = timestamp if timestamp else datetime.now().isoformat()
        self.user_prompt = user_prompt
        self.actual_output = actual_output
        self.target_output = target_output
        self.intent = intent
        self.parent_id = parent_id
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "user_prompt": self.user_prompt,
            "actual_output": self.actual_output,
            "target_output": self.target_output,
            "intent": self.intent,
            "parent_id": self.parent_id,
            "metadata": json.dumps(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Interaction':
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            timestamp=data["timestamp"],
            user_prompt=data["user_prompt"],
            actual_output=data["actual_output"],
            target_output=data["target_output"],
            intent=data.get("intent"),
            parent_id=data["parent_id"],
            metadata=json.loads(data["metadata"]) if data["metadata"] else {}
        )