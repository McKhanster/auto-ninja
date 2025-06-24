# app/models/agent.py
from datetime import datetime
import json

class Agent:
    def __init__(self, id: int = None, name: str = "", role: str = "", role_info: dict = None, tool_ids: str = "", created_at: str = None, updated_at: str = None, metadata: dict = None):
        self.id = id
        self.name = name
        self.role = role
        self.role_info = role_info
        self.tool_ids = tool_ids  # Comma-separated string of tool IDs
        self.created_at = created_at if created_at else datetime.now().isoformat()
        self.updated_at = updated_at
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "role_info": json.dumps(self.role_info) if self.role_info else None,
            "tool_ids": self.tool_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": json.dumps(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Agent':
        return cls(
            id=data["id"],
            name=data["name"],
            role=data["role"],
            role_info=json.loads(data["role_info"]) if data["role_info"] else None,
            tool_ids=data.get("tool_ids", ""),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=json.loads(data["metadata"]) if data["metadata"] else {}
        )