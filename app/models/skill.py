# app/models/skill.py
from datetime import datetime
import json

class Skill:
    def __init__(self, id: int = None, agent_id: int = None, name: str = "", description: str = None, instructions: dict = None, proficiency: float = None, script_id: int = None, acquired: bool = False, created_at: str = None, updated_at: str = None, metadata: dict = None):
        self.id = id
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.instructions = instructions if instructions is not None else {"type": "text", "content": "", "parameters": []}
        self.proficiency = proficiency
        self.script_id = script_id
        self.acquired = acquired
        self.created_at = created_at if created_at else datetime.now().isoformat()
        self.updated_at = updated_at
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "proficiency": self.proficiency,
            "script_id": self.script_id,
            "acquired": self.acquired,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": json.dumps(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Skill':
        return cls(
            id=data["id"],
            agent_id=data.get("agent_id"),
            name=data["name"],
            description=data["description"],
            instructions=data["instructions"],
            proficiency=data["proficiency"],
            script_id=data["script_id"],
            acquired=data.get("acquired", False),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=json.loads(data["metadata"]) if data["metadata"] else {}
        )