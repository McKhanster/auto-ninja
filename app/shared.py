from app.scripting.agent_manager import AgentManager
from .inference.engine import AutoNinjaInference
from .models.agent import Agent
from .models.tool import Tool
from sentence_transformers import SentenceTransformer
import torch
import logging

logger = logging.getLogger(__name__)

class SingletonSentenceTransformer:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            logger.info(f"Initialized singleton SentenceTransformer on {device}")
        return cls._instance

class SingletonInference:
    _instance = None
    _agent = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = AutoNinjaInference()
            memory_manager = cls._instance.memory_manager
            agent_manager = AgentManager(cls._instance, memory_manager)
            
            try:
                agent_data = memory_manager.memory.sqlite_store.cursor.execute(
                    "SELECT id, name, role FROM agents LIMIT 1"
                ).fetchone()
                if agent_data:
                    agent = Agent(id=agent_data[0], name=agent_data[1], role=agent_data[2])
                    memory_manager.set_current_agent(agent)
                    logger.info(f"Loaded existing agent: {agent.name} (ID: {agent.id}, Role: {agent.role})")
                else:
                    name = input("Enter the agent's name: ").strip()
                    role = input("Enter the agent's role: ").strip()
                    agent = Agent(name=name, role=role, tool_ids="")
                    agent.role_info = cls._instance.get_role_info(role)
                    memory_manager.memory.store_agent(agent)
                    memory_manager.set_current_agent(agent)
                    for tool_name in agent.role_info.get("tools", []):
                        tool = Tool(
                            agent_id=agent.id,
                            name=tool_name,
                            description=f"Tool suggested for {role} role",
                            instructions=None,
                            parameters=None,
                            usage_frequency=0,
                            venv_path=None,
                            dependencies=[]
                        )
                        tool_id = memory_manager.memory.store_tool(tool)
                        agent.tool_ids = f"{tool_id}" if not agent.tool_ids else f"{agent.tool_ids},{tool_id}"
                        memory_manager.memory.sqlite_store.cursor.execute(
                            'INSERT OR IGNORE INTO agent_tools (agent_id, tool_id) VALUES (?, ?)',
                            (agent.id, tool_id)
                        )
                    memory_manager.memory.sqlite_store.conn.commit()
                    agent_manager.auto_acquire_skills(role)
                    logger.info(f"Created new agent: {agent.name} (ID: {agent.id}, Role: {agent.role})")
            except Exception as e:
                logger.error(f"Failed to create or load agent: {e}", exc_info=True)
                agent = Agent(name="DefaultAgent", role="General", tool_ids="")
                memory_manager.memory.store_agent(agent)
                memory_manager.set_current_agent(agent)
                agent_manager.auto_acquire_skills("General")
                logger.info("Fallback to default agent: DefaultAgent (Role: General)")
            cls._agent = agent
        return cls._instance

inference_engine = SingletonInference.get_instance()