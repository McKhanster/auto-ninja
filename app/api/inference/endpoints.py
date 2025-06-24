from fastapi import APIRouter, HTTPException
from typing import Dict, List
from app.scripting.agent_manager import AgentManager
from ...shared import inference_engine
from ...middleware.memory import MemoryManager
from ...models.agent import Agent
from ...models.inference_request import InferenceRequest
from ...models.inference_response import InferenceResponse
import logging

router = APIRouter(prefix="/inference", tags=["inference"])
memory_manager = MemoryManager(sqlite_path="memory.db", qdrant_path="qdrant.db")
agent_manager = AgentManager(inference_engine, memory_manager)

@router.post("/agent/create")
async def create_agent(request: Dict[str, str]) -> Dict[str, str]:
    name = request.get("name")
    role = request.get("role")
    if not name or not role:
        raise HTTPException(status_code=400, detail="Name and role are required.")
    agent = Agent(name=name, role=role)
    agent.role_info = inference_engine.get_role_info(role)
    inference_engine.memory_manager.memory.store_agent(agent)
    inference_engine.memory_manager.set_current_agent(agent)
    return {"agent_id": str(agent.id), "name": name, "role": role}

@router.get("/agent/{agent_id}")
async def get_agent(agent_id: int) -> Dict:
    agent = memory_manager.memory.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"agent_id": str(agent.id), "name": agent.name, "role": agent.role, "created_at": agent.created_at}

@router.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest) -> InferenceResponse:
    result = inference_engine.process(request.message, script_manager=agent_manager)
    return InferenceResponse(prediction=result)

@router.get("/interaction/{interaction_id}")
async def get_interaction(interaction_id: int) -> Dict:
    interaction = memory_manager.get_interaction(interaction_id)
    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")
    return {"id": interaction.id, "user_prompt": interaction.user_prompt, "actual_output": interaction.actual_output}

@router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}