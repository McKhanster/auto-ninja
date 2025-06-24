# app/models/tool.py
from datetime import datetime
import json

class Tool:
    def __init__(self, id: int = None, agent_id: int = None, name: str = "", description: str = None, instructions: dict = None, parameters: list = None, usage_frequency: int = None, venv_path: str = None, dependencies: list = None, created_at: str = None, updated_at: str = None, metadata: dict = None):
        self.id = id
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.instructions = instructions
        self.parameters = parameters if parameters is not None else []  # List of dicts: {"name", "type", "description", "required", "default", "fields" (for objects)}
        self.usage_frequency = usage_frequency
        self.venv_path = venv_path
        self.dependencies = dependencies if dependencies else []
        self.created_at = created_at if created_at else datetime.now().isoformat()
        self.updated_at = updated_at
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "instructions": json.dumps(self.instructions) if self.instructions else None,
            "parameters": json.dumps(self.parameters) if self.parameters else None,
            "usage_frequency": self.usage_frequency,
            "venv_path": self.venv_path,
            "dependencies": json.dumps(self.dependencies) if self.dependencies else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": json.dumps(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Tool':
        return cls(
            id=data["id"],
            agent_id=data.get("agent_id"),
            name=data["name"],
            description=data["description"],
            instructions=json.loads(data["instructions"]) if data["instructions"] else None,
            parameters=json.loads(data["parameters"]) if data["parameters"] else None,
            usage_frequency=data["usage_frequency"],
            venv_path=data["venv_path"],
            dependencies=json.loads(data["dependencies"]) if data["dependencies"] else None,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=json.loads(data["metadata"])
        )