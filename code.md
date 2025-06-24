
==================================================
File: shared.py
==================================================

# app/shared.py
from app.scripting.agent_manager import AgentManager
from .inference.engine import AutoNinjaInference
from .models.agent import Agent
from .models.tool import Tool
import logging

logger = logging.getLogger(__name__)

class SingletonInference:
    _instance = None
    _agent = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = AutoNinjaInference()
            memory_manager = cls._instance.memory_manager
            agent_manager = AgentManager(cls._instance, memory_manager)  # Updated
            
            agent_data = memory_manager.memory.sqlite_store.cursor.execute(
                "SELECT id, name, role FROM agents LIMIT 1"
            ).fetchone()
            if agent_data:
                agent = Agent(id=agent_data[0], name=agent_data[1], role=agent_data[2])
                memory_manager.set_current_agent(agent)
                logger.info(f"Loaded existing agent: {agent.name} (ID: {agent.id}, Role: {agent.role})")
            else:
                try:
                    name = input("Enter the agent's name: ").strip()
                    role = input("Enter the agent's role: ").strip()
                    agent = Agent(name=name, role=role)
                    agent.role_info = cls._instance.get_role_info(role)
                    memory_manager.memory.store_agent(agent)
                    memory_manager.set_current_agent(agent)
                    for tool_name in agent.role_info.get("tools", []):
                        tool = Tool(
                            agent_id=agent.id,
                            name=tool_name,
                            description=f"Tool suggested for {role} role",
                            instructions=None,
                            usage_frequency=0,
                            venv_path=None,
                            dependencies=[]
                        )
                        memory_manager.memory.store_tool(tool)
                    agent_manager.auto_acquire_skills(role)  # Updated
                    logger.info(f"Created new agent: {agent.name} (ID: {agent.id}, Role: {agent.role})")
                except Exception as e:
                    logger.error(f"Failed to create agent or acquire skills: {str(e)}", exc_info=True)
                    agent = Agent(name="DefaultAgent", role="General")
                    memory_manager.memory.store_agent(agent)
                    memory_manager.set_current_agent(agent)
                    agent_manager.auto_acquire_skills("General")  # Updated
            cls._agent = agent
        return cls._instance

inference_engine = SingletonInference.get_instance()
memory_manager = inference_engine.memory_manager

==================================================
File: main.py
==================================================

import logging
from importlib.metadata import version
from fastapi import FastAPI
from .api.inference.endpoints import router as inference_router
from .shared import inference_engine  # Import singleton
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting Auto Ninja with the following package versions:")
logger.info(f"torch: {version('torch')}")
logger.info(f"transformers: {version('transformers')}")
logger.info(f"torchvision: {version('torchvision')}")
logger.info(f"sentence-transformers: {version('sentence-transformers')}")
logger.info(f"numpy: {version('numpy')}")
logger.info("Pre-loading inference engine model...")
# Model is loaded here via singleton import
logger.info("Inference engine model pre-loaded")
# Log new speech-related dependencies
try:
    logger.info(f"openai-whisper: {version('openai-whisper')}")
    logger.info(f"gtts: {version('gtts')}")
    logger.info(f"pyttsx3: {version('pyttsx3')}")
    logger.info(f"pyaudio: {version('pyaudio')}")
    logger.info(f"pydub: {version('pydub')}")
except Exception as e:
    logger.warning(f"Could not log version for some speech dependencies: {str(e)}")

app = FastAPI()
app.include_router(inference_router)

==================================================
File: api/inference/endpoints.py
==================================================

# app/api/inference/endpoints.py
from fastapi import APIRouter, HTTPException
from typing import Dict, List

from app.scripting.agent_manager import AgentManager
from ...shared import inference_engine, memory_manager
from ...models.agent import Agent
from ...models.inference_request import InferenceRequest
from ...models.inference_response import InferenceResponse
import logging

router = APIRouter(prefix="/inference", tags=["inference"])
agent_manager = AgentManager(inference_engine, memory_manager)  # Updated

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
    result = inference_engine.process(request.message, script_manager=agent_manager)  # Updated
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

==================================================
File: hybrid_memory/sqlite_store.py
==================================================

import sqlite3
import json
from typing import List
from ..models.interaction import Interaction
from ..models.agent import Agent
from ..models.skill import Skill
from ..models.tool import Tool

class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def initialize(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id INTEGER,
                timestamp TEXT,
                user_prompt TEXT,
                actual_output TEXT,
                target_output TEXT,
                parent_id INTEGER,
                metadata TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(id),
                FOREIGN KEY (parent_id) REFERENCES interactions(id)
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                role TEXT,
                role_info TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id INTEGER,
                name TEXT,
                description TEXT,
                instructions TEXT,
                proficiency REAL,
                script_id INTEGER,
                acquired INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id INTEGER,
                name TEXT,
                description TEXT,
                instructions TEXT,
                usage_frequency INTEGER,
                venv_path TEXT,
                dependencies TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_tools (
                agent_id INTEGER,
                tool_id INTEGER,
                PRIMARY KEY (agent_id, tool_id),
                FOREIGN KEY (agent_id) REFERENCES agents(id),
                FOREIGN KEY (tool_id) REFERENCES tools(id)
            )
        ''')

        # Drop agent_skills if it exists
        self.cursor.execute('DROP TABLE IF EXISTS agent_skills')

        self.conn.commit()

    def store_interaction(self, interaction: Interaction) -> int:
        data = interaction.to_dict()
        self.cursor.execute('''
            INSERT INTO interactions (agent_id, timestamp, user_prompt, actual_output, target_output, parent_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data["agent_id"],
            data["timestamp"],
            data["user_prompt"],
            data["actual_output"],
            data["target_output"],
            data["parent_id"],
            data["metadata"]
        ))
        interaction_id = self.cursor.lastrowid
        self.conn.commit()
        interaction.id = interaction_id
        return interaction_id

    def get_interaction(self, interaction_id: int) -> Interaction:
        self.cursor.execute('SELECT * FROM interactions WHERE id = ?', (interaction_id,))
        data = self.cursor.fetchone()
        if data:
            return Interaction.from_dict({
                "id": data[0],
                "agent_id": data[1],
                "timestamp": data[2],
                "user_prompt": data[3],
                "actual_output": data[4],
                "target_output": data[5],
                "parent_id": data[6],
                "metadata": data[7]
            })
        return None

    def get_interactions_by_ids(self, interaction_ids: list[int]) -> list[Interaction]:
        if not interaction_ids:
            return []
        placeholders = ','.join('?' for _ in interaction_ids)
        self.cursor.execute(f'SELECT * FROM interactions WHERE id IN ({placeholders})', interaction_ids)
        rows = self.cursor.fetchall()
        return [Interaction.from_dict({
            "id": row[0],
            "agent_id": row[1],
            "timestamp": row[2],
            "user_prompt": row[3],
            "actual_output": row[4],
            "target_output": row[5],
            "parent_id": row[6],
            "metadata": row[7]
        }) for row in rows]

    def store_agent(self, agent: Agent) -> int:
        data = agent.to_dict()
        self.cursor.execute('''
            INSERT INTO agents (name, role, role_info, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data["name"],
            data["role"],
            data["role_info"],
            data["created_at"],
            data["updated_at"],
            data["metadata"]
        ))
        agent_id = self.cursor.lastrowid
        agent.id = agent_id

        # Store tool relationships (skill_ids no longer needed here)
        if agent.tool_ids:
            tool_ids = [int(tid) for tid in agent.tool_ids.split(",") if tid]
            for tool_id in tool_ids:
                self.cursor.execute('INSERT INTO agent_tools (agent_id, tool_id) VALUES (?, ?)', (agent_id, tool_id))

        self.conn.commit()
        return agent_id

    def get_agent(self, agent_id: int) -> Agent:
        self.cursor.execute('SELECT * FROM agents WHERE id = ?', (agent_id,))
        data = self.cursor.fetchone()
        if data:
            agent = Agent.from_dict({
                "id": data[0],
                "name": data[1],
                "role": data[2],
                "role_info": data[3],
                "created_at": data[4],
                "updated_at": data[5],
                "metadata": data[6]
            })
            # Populate tool_ids
            self.cursor.execute('SELECT tool_id FROM agent_tools WHERE agent_id = ?', (agent_id,))
            tool_ids = [row[0] for row in self.cursor.fetchall()]
            agent.tool_ids = ",".join(str(tid) for tid in tool_ids) if tool_ids else ""
            return agent
        return None

    def store_skill(self, skill: Skill) -> int:
        data = skill.to_dict()
        # Serialize instructions to JSON string
        instructions_json = json.dumps(data["instructions"]) if data["instructions"] is not None else None
        self.cursor.execute('''
            INSERT INTO skills (agent_id, name, description, instructions, proficiency, script_id, acquired, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data["agent_id"],
            data["name"],
            data["description"],
            instructions_json,  # Use serialized JSON
            data["proficiency"],
            data["script_id"],
            1 if data["acquired"] else 0,
            data["created_at"],
            data["updated_at"],
            data["metadata"]
        ))
        skill_id = self.cursor.lastrowid
        self.conn.commit()
        skill.id = skill_id
        return skill_id

    def get_skill(self, skill_id: int) -> Skill:
        self.cursor.execute('SELECT * FROM skills WHERE id = ?', (skill_id,))
        data = self.cursor.fetchone()
        if data:
            return Skill.from_dict({
                "id": data[0],
                "agent_id": data[1],
                "name": data[2],
                "description": data[3],
                "instructions": json.loads(data[4]) if data[4] else None,
                "proficiency": data[5],
                "script_id": data[6],
                "acquired": bool(data[7]),
                "created_at": data[8],
                "updated_at": data[9],
                "metadata": json.loads(data[10]) if data[10] else {}
            })
        return None

    def store_tool(self, tool: Tool) -> int:
        data = tool.to_dict()
        self.cursor.execute('''
            INSERT INTO tools (agent_id, name, description, instructions, usage_frequency, venv_path, dependencies, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data["agent_id"],
            data["name"],
            data["description"],
            data["instructions"],
            data["usage_frequency"],
            data["venv_path"],
            data["dependencies"],
            data["created_at"],
            data["updated_at"],
            data["metadata"]
        ))
        tool_id = self.cursor.lastrowid
        self.conn.commit()
        tool.id = tool_id
        return tool_id

    def get_tool(self, tool_id: int) -> Tool:
        self.cursor.execute('SELECT * FROM tools WHERE id = ?', (tool_id,))
        data = self.cursor.fetchone()
        if data:
            return Tool.from_dict({
                "id": data[0],
                "agent_id": data[1],
                "name": data[2],
                "description": data[3],
                "instructions": data[4],
                "usage_frequency": data[5],
                "venv_path": data[6],
                "dependencies": data[7],
                "created_at": data[8],
                "updated_at": data[9],
                "metadata": data[10]
            })
        return None
    
    def get_skills_by_agent(self, agent_id: int) -> List[Skill]:
        """Retrieve all skills for a given agent_id from the skills table."""
        query = "SELECT id, agent_id, name, description, instructions, script_id, acquired, created_at, updated_at, metadata FROM skills WHERE agent_id = ?"
        self.cursor.execute(query, (agent_id,))
        rows = self.cursor.fetchall()

        skills = []
        for row in rows:
            skill = Skill(
                id=row[0],
                agent_id=row[1],
                name=row[2],
                description=row[3],
                instructions=json.loads(row[4]) if row[4] else {"type": "text", "content": ""},
                script_id=row[5],
                acquired=bool(row[6]),
                created_at=row[7],
                updated_at=row[8],
                metadata=json.loads(row[9]) if row[9] else {}
            )
            skills.append(skill)
        return skills

==================================================
File: hybrid_memory/core.py
==================================================

# app/hybrid_memory/core.py
from .qdrant_store import QdrantStore
from ..models.interaction import Interaction
from ..models.agent import Agent
from ..models.skill import Skill
from ..models.tool import Tool
from .utils.embedding import generate_embedding
from .utils.summarizer import summarize_interaction
from .utils.clustering import cluster_embeddings

class HybridMemory:
    def __init__(self, sqlite_path: str, qdrant_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        from .sqlite_store import SQLiteStore
        self.sqlite_store = SQLiteStore(sqlite_path)
        self.qdrant_store = QdrantStore(qdrant_path, embedding_model)

    def initialize(self):
        self.sqlite_store.initialize()
        self.qdrant_store.initialize()

    def store_interaction(self, interaction: Interaction) -> int:
        interaction_id = self.sqlite_store.store_interaction(interaction)
        summary = summarize_interaction(interaction)
        embedding = generate_embedding(summary)
        self.qdrant_store.store_summary(interaction, summary, embedding)
        return interaction_id

    def get_interaction(self, interaction_id: int) -> Interaction:
        return self.sqlite_store.get_interaction(interaction_id)

    def get_similar_interactions(self, prompt: str, limit: int = 3) -> list[Interaction]:
        embedding = generate_embedding(prompt)
        similar_summaries = self.qdrant_store.get_similar_summaries(embedding, limit=limit)
        interaction_ids = [summary["id"] for summary in similar_summaries]
        return self.sqlite_store.get_interactions_by_ids(interaction_ids)

    def cluster_interactions(self, n_clusters: int = 5):
        embeddings, ids = self.qdrant_store.get_all_embeddings()
        if not embeddings:
            return
        cluster_ids = cluster_embeddings(embeddings, n_clusters)
        for interaction_id, cluster_id in zip(ids, cluster_ids):
            self.qdrant_store.update_cluster_id(interaction_id, f"cluster_{cluster_id}")

    def store_agent(self, agent: Agent) -> int:
        return self.sqlite_store.store_agent(agent)

    def get_agent(self, agent_id: int) -> Agent:
        return self.sqlite_store.get_agent(agent_id)

    def store_skill(self, skill: Skill) -> int:
        return self.sqlite_store.store_skill(skill)

    def get_skill(self, skill_id: int) -> Skill:
        return self.sqlite_store.get_skill(skill_id)

    def store_tool(self, tool: Tool) -> int:
        return self.sqlite_store.store_tool(tool)

    def get_tool(self, tool_id: int) -> Tool:
        return self.sqlite_store.get_tool(tool_id)

==================================================
File: hybrid_memory/qdrant_store.py
==================================================

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from ..models.interaction import Interaction
import logging

logger = logging.getLogger(__name__)

class QdrantStore:
    def __init__(self, qdrant_path: str, embedding_model: str):
        # Use standalone server instead of embedded mode
        self.client = QdrantClient(location="localhost:6333")
        self.embedding_model = embedding_model
        self.default_collection = "interaction_summaries"

    def initialize(self):
        """Create default and skill collections if they don’t exist."""
        for collection_name in [self.default_collection, "skill_summaries"]:
            try:
                self.client.get_collection(collection_name)
                logger.info(f"Collection {collection_name} already exists")
            except Exception:
                logger.info(f"Creating collection {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                logger.info(f"Collection {collection_name} created successfully")

    def store_summary(self, obj: object, summary: str, embedding: list[float], collection_name: str = None):
        collection = collection_name or self.default_collection
        # Flexibly handle timestamp or created_at, with a fallback
        timestamp = getattr(obj, "timestamp", getattr(obj, "created_at", "unknown"))
        point = PointStruct(
            id=obj.id,
            vector=embedding,
            payload={
                "id": obj.id,
                "agent_id": obj.agent_id,
                "timestamp": timestamp,
                "summary": summary,
                "category": "skill" if collection == "skill_summaries" else "general",
                "metadata": obj.metadata
            }
        )
        self.client.upsert(collection_name=collection, points=[point])
        logger.info(f"Stored summary for {collection} ID {obj.id} in Qdrant")

    def get_similar_summaries(self, embedding: list[float], limit: int = 3, collection_name: str = None) -> list[dict]:
        collection = collection_name or self.default_collection
        search_result = self.client.search(
            collection_name=collection,
            query_vector=embedding,
            limit=limit
        )
        return [hit.payload for hit in search_result]

    def get_all_embeddings(self) -> tuple[list[list[float]], list[int]]:
        points = self.client.scroll(
            collection_name=self.collection_name,
            with_vectors=True,
            limit=10000
        )[0]
        embeddings = [point.vector for point in points]
        ids = [point.id for point in points]
        return embeddings, ids

    def update_cluster_id(self, interaction_id: int, cluster_id: str):
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"cluster_id": cluster_id},
            points=[interaction_id]
        )

==================================================
File: hybrid_memory/utils/clustering.py
==================================================

from sklearn.cluster import KMeans

def cluster_embeddings(embeddings: list[list[float]], n_clusters: int) -> list[int]:
    if len(embeddings) < n_clusters:
        n_clusters = max(1, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels.tolist()

==================================================
File: hybrid_memory/utils/embedding.py
==================================================

from sentence_transformers import SentenceTransformer

def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    model = SentenceTransformer(model_name)
    embedding = model.encode(text, convert_to_numpy=True).tolist()
    return embedding

==================================================
File: hybrid_memory/utils/summarizer.py
==================================================



from app.models.interaction import Interaction


def summarize_interaction(interaction: Interaction) -> str:
    user_part = f"User: {interaction.user_prompt if interaction.user_prompt else '[No prompt provided]'}"
    agent_part = f"Agent: {interaction.target_output or interaction.actual_output}"
    return f"{user_part} | {agent_part}"

==================================================
File: inference/__init__.py
==================================================



==================================================
File: inference/engine.py
==================================================

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
            response = requests.post(self.grok_url, json=payload.model_dump(), headers=self.grok_headers, timeout=15)
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

==================================================
File: config/settings.py
==================================================

from pathlib import Path
from dotenv import load_dotenv
import os
import requests

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Settings:
    MODEL_NAME: str = os.getenv("MODEL_NAME", "google/gemma-3-1b-it")
    DEVICE_TYPE: str = os.getenv("DEVICE_TYPE", "cpu").lower()
    MAX_SEQUENCE_LENGTH: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "128"))
    HF_TOKEN: str = os.getenv("HF_TOKEN", None)
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "50"))
    XAI_API_KEY: str = os.getenv("XAI_API_KEY", None)
    GROK_MODEL: str = "grok-3-beta"
    CHAT_COMPLETION_URL: str = "https://api.x.ai/v1/chat/completions"
    IMAGE_GENERATION_URL = str(os.getenv("IMAGE_GENERATION_URL", "https://api.x.ai/v1/image/generations"))
    GROK_URL = CHAT_COMPLETION_URL
    VALID_DEVICES = {"cpu", "gpu", "cuda"}
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "tiny")
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", None)
    SPEECH_DIR: str = os.getenv("SPEECH_DIR", "speech/")
    REQUEST_TIMEOUT: float = float(os.getenv("REQUEST_TIMEOUT", "30"))  # Default 30s

    def __init__(self):
        if self.DEVICE_TYPE not in self.VALID_DEVICES:
            raise ValueError(f"Invalid DEVICE_TYPE: {self.DEVICE_TYPE}. Must be one of {self.VALID_DEVICES}")
        if not self.XAI_API_KEY:
            print("Warning: XAI_API_KEY not set; Grok API integration will be disabled")
        if "llama" in self.MODEL_NAME.lower() and not self.HF_TOKEN:
            print("Warning: HF_TOKEN might be required for LLaMA models")
        Path(self.SPEECH_DIR).mkdir(exist_ok=True)
        if not self.TTS_ENGINE:
            self.TTS_ENGINE = "gTTS" if self.check_connectivity() and self.XAI_API_KEY else "pyttsx3"
            print(f"TTS_ENGINE set to {self.TTS_ENGINE}")

    def check_connectivity(self) -> bool:
        try:
            response = requests.get("https://google.com", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

settings = Settings()

==================================================
File: config/constants.py
==================================================

# Python 3.12 built-in modules (subset, full list at docs.python.org/3/py-modindex.html)
BUILT_IN_PACKAGES = [
    "os", "sys", "json", "time", "datetime", "math", "random", "re", "subprocess",
    "pathlib", "io", "threading", "queue", "logging", "functools", "itertools",
    "collections", "shutil", "glob", "fnmatch", "tempfile", "uuid", "hashlib",
    "base64", "binascii", "struct", "zlib", "gzip", "bz2", "zipfile", "tarfile",
    "socket", "select", "signal", "stat", "errno", "platform", "getpass", "ctypes"
]

==================================================
File: config/__init__.py
==================================================



==================================================
File: speech/tts.py
==================================================

from gtts import gTTS
import pyttsx3
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(self, engine: str = settings.TTS_ENGINE):
        """Initialize TTS engine (gTTS or pyttsx3)."""
        self.engine = engine
        logger.info(f"Initializing TTS with engine: {engine}")
        if self.engine == "pyttsx3":
            self.tts = pyttsx3.init()
            self.tts.setProperty("rate", 150)  # Speed adjustment
            self.tts.setProperty("volume", 1.0)  # Max volume

    def synthesize(self, text: str, output_path: str) -> None:
        """Convert text to speech and save to output_path."""
        try:
            logger.info(f"Synthesizing text: {text}")
            if self.engine == "gTTS":
                tts = gTTS(text=text, lang="en", slow=False)
                tts.save(output_path)  # Saves as MP3
            elif self.engine == "pyttsx3":
                self.tts.save_to_file(text, output_path)  # Saves as WAV
                self.tts.runAndWait()
            logger.info(f"Audio saved to {output_path}")
        except Exception as e:
            logger.error(f"TTS synthesis failed: {str(e)}")
            raise ValueError(f"Failed to synthesize audio: {str(e)}")

==================================================
File: speech/stt.py
==================================================

import whisper
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)

class SpeechToText:
    def __init__(self, model_name: str = settings.WHISPER_MODEL):
        """Initialize Whisper model for STT."""
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        self.options = {"fp16": False}  # Use FP32 for broader compatibility

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text using Whisper."""
        try:
            logger.info(f"Transcribing audio from {audio_path}")
            result = self.model.transcribe(audio_path, **self.options)
            text = result["text"].strip()
            logger.info(f"Transcription result: {text}")
            return text
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise ValueError(f"Failed to transcribe audio: {str(e)}")

==================================================
File: speech/utils.py
==================================================

import pyaudio
import wave
import os
from pydub import AudioSegment
from pydub.playback import play
from ..config.settings import settings
import logging
import threading
import queue

logger = logging.getLogger(__name__)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def record_audio(output_path: str) -> None:
    """Record audio until Enter is pressed in the terminal."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                       input=True, frames_per_buffer=CHUNK)
    
    logger.info(f"Recording audio to {output_path}... Press Enter to stop.")
    frames = []
    stop_event = queue.Queue()  # Thread-safe way to signal stop

    def wait_for_enter():
        input()  # Blocks until Enter is pressed
        stop_event.put(True)

    # Start a thread to wait for Enter
    input_thread = threading.Thread(target=wait_for_enter)
    input_thread.start()

    # Record until stop signal is received
    while stop_event.qsize() == 0:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except IOError as e:
            logger.warning(f"IOError during recording: {str(e)}")
            continue

    logger.info("Recording stopped by user (Enter pressed).")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    input_thread.join()  # Clean up the thread

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def play_audio(audio_path: str) -> None:
    """Play audio file using pydub."""
    try:
        logger.info(f"Playing audio from {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        play(audio)
    except Exception as e:
        logger.error(f"Playback failed: {str(e)}")
        raise ValueError(f"Failed to play audio: {str(e)}")

def cleanup_files(*paths: str) -> None:
    """Delete temporary audio files."""
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted temporary file: {path}")

==================================================
File: middleware/memory.py
==================================================

# app/middleware/memory.py
from ..hybrid_memory.core import HybridMemory
from ..models.interaction import Interaction
from ..models.agent import Agent
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, sqlite_path: str, qdrant_path: str):
        self.memory = HybridMemory(
            sqlite_path=sqlite_path,  # Renamed to sqlite_path
            qdrant_path=qdrant_path,
            embedding_model="all-MiniLM-L6-v2"
        )
        self.memory.initialize()
        self.current_agent = None

    def set_current_agent(self, agent: Agent):
        self.current_agent = agent
        logger.info(f"Current agent set to: {agent.name} (ID: {agent.id})")

    def process(self, user_prompt: str, local_output: str, final_output: str) -> str:
        if not self.current_agent:
            raise ValueError("No current agent set. Please create or select an agent first.")

        interaction = Interaction(
            agent_id=self.current_agent.id,
            user_prompt=user_prompt,
            actual_output=local_output,
            target_output=final_output,
            parent_id=None,
            metadata={"user_id": "user123", "session_id": "session456", "parent_id": None}
        )

        self.memory.store_interaction(interaction)
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
            # Extract skill names from list of dicts or handle as strings
            skills = [skill["name"] if isinstance(skill, dict) else str(skill) for skill in self.current_agent.role_info.get('skills', [])]
            context += f"Skills: {', '.join(skills)}\n"
            context += f"Tools: {', '.join(self.current_agent.role_info.get('tools', []))}\n"
        return context

    def learn(self):
        print("Learning not yet implemented in this minimal version.")

==================================================
File: scripting/utils.py
==================================================

# app/scripting/utils.py
import re
import os
from pathlib import Path
from .constants import SCRIPT_DIR, FILE_PATTERN
from ..inference.engine import AutoNinjaInference
import json
from ..config.constants import BUILT_IN_PACKAGES

def generate_script_content(inference_engine: AutoNinjaInference, prompt: str, skill_name: str = None) -> str:
    """Generate Python script content from a prompt."""
    code_prompt = f"Output only the Python code to {prompt}. Use sys.argv[1] for parameters if applicable."
    content = inference_engine.process(code_prompt)
    content = re.sub(r"```python\s*(.*?)\s*```", r"\1", content, flags=re.DOTALL)
    if not content.strip() or not any(kw in content.lower() for kw in ["print", "def", "import", "return"]):
        content = f"import sys\nprint('Script for {skill_name or 'task'} executed with:', sys.argv[1] if len(sys.argv) > 1 else 'no argument')"
    return content.strip()

def save_script(file_name: str, content: str) -> bool:
    try:
        SCRIPT_DIR.mkdir(exist_ok=True)
        with open(SCRIPT_DIR / file_name, "w") as f:
            f.write(content)
        return True
    except Exception:
        return False

def load_script(file_name: str) -> str:
    if not script_exists(file_name):
        raise FileNotFoundError(f"Script {file_name} not found")
    with open(SCRIPT_DIR / file_name, "r") as f:
        return f.read()

def delete_script(file_name: str) -> bool:
    if not script_exists(file_name):
        return False
    os.remove(SCRIPT_DIR / file_name)
    return True

def script_exists(file_name: str) -> bool:
    return (SCRIPT_DIR / file_name).exists()

def generate_file_name(category: str, task: str) -> str:
    pattern = f"{category}_{task}_\\d{{3}}\\.py"
    existing_files = [f.name for f in SCRIPT_DIR.glob("*.py") if re.match(pattern, f.name)]
    next_num = len(existing_files) + 1
    return f"{category}_{task}_{next_num:03d}.py"

def extract_file_name(prompt: str) -> str | None:
    match = re.search(r"(?:run|update|delete)\s+code\s+([a-zA-Z]+_[a-zA-Z0-9]+_\d{3}\.py)", prompt, re.IGNORECASE)
    return match.group(1) if match else None

def validate_file_name(file_name: str) -> bool:
    return bool(re.match(FILE_PATTERN, file_name))

def infer_dependencies(code: str, inference_engine: AutoNinjaInference) -> tuple[list[str], list[str]]:
    """Infer pip packages and system dependencies from Python code."""
    imports = []
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("import "):
            module = line.split()[1].split(".")[0]
            imports.append(module)
        elif line.startswith("from "):
            parts = line.split()
            if len(parts) > 1:
                module = parts[1].split(".")[0]
                imports.append(module)
    imports = list(set(imports) - set(BUILT_IN_PACKAGES))
    
    if not imports:
        return [], []
    
    prompt = (
        f"For Python imports: {', '.join(imports)}, provide exact pip-installable package names and system dependencies "
        "as JSON: {{'pip_packages': list[str], 'system_deps': list[str]}}. Exclude standard library modules."
    )
    response = inference_engine.process(prompt)
    try:
        data = robust_parse_json(response)
        return data.get("pip_packages", imports), data.get("system_deps", [])
    except ValueError:
        return imports, []

def robust_parse_json(raw_response: str) -> dict:
    """Parse JSON from a response, handling malformed or truncated data."""
    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    json_str = json_match.group(0) if json_match else raw_response
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str += '}' if not json_str.endswith('}') else ''
        return json.loads(json_str)

==================================================
File: scripting/constants.py
==================================================

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.parent / "scripts"
SCRIPT_DIR1 = Path(__file__).parent.parent.parent
DEFAULT_LANGUAGE = "python"
TRIGGERS = {
    "generate": "generate code",
    "execute": "run code",
    "update": "update code",
    "delete": "delete code"
}
FILE_PATTERN = r"(\w+)_(\w+)_(\d{3})\.py"

==================================================
File: scripting/agent_manager.py
==================================================

# app/scripting/manager.py (to be renamed to agent_manager.py)
from datetime import datetime
import getpass
import json
import os
import subprocess
from typing import List, Optional, Dict
from .constants import TRIGGERS, SCRIPT_DIR
from .utils import (
    save_script, load_script, delete_script, script_exists,
    generate_file_name, extract_file_name, validate_file_name,
    generate_script_content, infer_dependencies, robust_parse_json
)
from ..inference.engine import AutoNinjaInference
from ..middleware.memory import MemoryManager
from ..models.tool import Tool
from ..models.skill import Skill
import logging
import re

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self, inference_engine: AutoNinjaInference, memory_manager: MemoryManager):
        self.inference_engine = inference_engine
        self.memory_manager = memory_manager
        self.pending_action: Optional[str] = None
        self.pending_details: Dict = {}
        logger.info("AgentManager initialized")

    def is_script_prompt(self, prompt: str) -> Optional[str]:
        prompt_lower = prompt.lower()
        for action, trigger in TRIGGERS.items():
            if trigger in prompt_lower:
                logger.info(f"Detected script prompt: {action}")
                return action
        return None

    def request_confirmation(self, action: str, details: dict) -> str:
        self.pending_action = action
        self.pending_details = details
        action_msg = "this code" if action == "generate" else f"{details.get('file_name', 'the code')}"
        return f"Do you want to {action} {action_msg}?"

    def confirm_action(self, user_response: str) -> Optional[str]:
        if user_response.lower() in ["yes", "y", "sure"]:
            action = self.pending_action
            details = self.pending_details
            self.pending_action = None
            self.pending_details = {}
            if action == "generate":
                return self.generate_script(details["prompt"])
            elif action == "execute":
                return self.execute_script(details["file_name"])
            elif action == "update":
                return self.update_script(details["prompt"], details["file_name"])
            elif action == "delete":
                return self.delete_script(details["file_name"])
        self.pending_action = None
        self.pending_details = {}
        return "Action cancelled."

    def generate_script(self, prompt: str, skill_name: str = None) -> str:
        content = generate_script_content(self.inference_engine, prompt, skill_name)
        parts = prompt.lower().replace("generate code", "").strip().split()
        category = parts[0] if parts and parts[0] not in {"for", "to", "please"} else "general"
        task = skill_name.replace(" ", "_") if skill_name else (parts[-1] if parts else "script")
        file_name = generate_file_name(category, task)
        save_script(file_name, content)

        tool = Tool(
            agent_id=self.memory_manager.current_agent.id,
            name=file_name,
            instructions={"type": "script", "path": f"scripts/{file_name}", "content": content},
            description=f"Generated script for {task}"
        )
        tool_id = self.memory_manager.memory.store_tool(tool)
        self.memory_manager.memory.sqlite_store.cursor.execute(
            "INSERT OR IGNORE INTO agent_tools (agent_id, tool_id) VALUES (?, ?)",
            (self.memory_manager.current_agent.id, tool_id)
        )
        self.memory_manager.memory.sqlite_store.conn.commit()

        summary = f"Generated a Python script: {file_name}"
        self.memory_manager.process(prompt, content, summary)
        return f"Code generated successfully! File: {file_name}"

    def execute_script(self, command: str) -> str:
        parts = command.split(" ", 1)
        file_name = parts[0].replace("scripts/", "")
        arg = parts[1] if len(parts) > 1 else ""
        script_path = SCRIPT_DIR / file_name

        if not script_path.exists():
            return "Invalid or missing script file."

        venv_path = "venvs/auto_ninja_venv"
        python_bin = os.path.join(venv_path, "bin" if os.name != "nt" else "Scripts", "python")
        content = load_script(file_name)
        pip_packages, system_deps = infer_dependencies(content, self.inference_engine)
        
        success, error_msg = self._setup_venv(venv_path, pip_packages, system_deps)
        if not success:
            return f"Error: {error_msg}"

        try:
            env = os.environ.copy()
            env["PATH"] = f"/usr/bin:/usr/local/bin:{env.get('PATH', '')}"
            cmd = [python_bin, str(script_path)] + ([arg] if arg else [])
            result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, env=env)
            return result.strip()
        except subprocess.CalledProcessError as e:
            correction_prompt = f"Fix this script that failed with error:\n{e.output}\nScript:\n{content}"
            correction_result = self.inference_engine.process(correction_prompt)
            if "```python" in correction_result:
                fixed_code = re.search(r"```python\s*(.*?)\s*```", correction_result, re.DOTALL).group(1)
                self.update_script(f"Update with fixed code: {fixed_code}", file_name)
                return self.execute_script(command)
            return f"Code execution failed: {e.output}"

    def update_script(self, prompt: str, file_name: str) -> str:
        if not validate_file_name(file_name) or not script_exists(file_name):
            return "Invalid or missing script file."
        
        content = generate_script_content(self.inference_engine, prompt.replace("update code", "").strip())
        save_script(file_name, content)

        tools = self.memory_manager.memory.get_tools_for_agent(self.memory_manager.current_agent)
        for tool in tools:
            if tool.name == file_name:
                tool.instructions["content"] = content
                tool.updated_at = datetime.now().isoformat()
                self.memory_manager.memory.store_tool(tool)
                break

        summary = f"Updated Python script: {file_name}"
        self.memory_manager.process(prompt, content, summary)
        return "Code updated successfully!"

    def delete_script(self, file_name: str) -> str:
        if not validate_file_name(file_name) or not script_exists(file_name):
            return "Invalid or missing script file."
        
        delete_script(file_name)
        tools = self.memory_manager.memory.get_tools_for_agent(self.memory_manager.current_agent)
        for tool in tools:
            if tool.name == file_name:
                self.memory_manager.memory.sqlite_store.cursor.execute(
                    "DELETE FROM agent_tools WHERE agent_id = ? AND tool_id = ?",
                    (self.memory_manager.current_agent.id, tool.id)
                )
                self.memory_manager.memory.sqlite_store.cursor.execute(
                    "DELETE FROM tools WHERE id = ?", (tool.id,)
                )
                self.memory_manager.memory.sqlite_store.conn.commit()
                break

        summary = f"Deleted Python script: {file_name}"
        self.memory_manager.process(f"delete code {file_name}", "Deleted", summary)
        return "Code deleted successfully!"

    def _setup_venv(self, venv_path: str, pip_packages: List[str], system_deps: List[str]) -> tuple[bool, str]:
        venv_python = os.path.join(venv_path, "bin" if os.name != "nt" else "Scripts", "python")
        venv_pip = os.path.join(venv_path, "bin" if os.name != "nt" else "Scripts", "pip")

        if not os.path.exists(venv_python):
            try:
                subprocess.check_call(["python", "-m", "venv", venv_path])
            except subprocess.CalledProcessError as e:
                return False, f"Failed to create venv: {e}"

        if pip_packages:
            try:
                installed = {line.split()[0].lower() for line in subprocess.check_output([venv_pip, "list"], text=True).splitlines()[2:]}
                to_install = [pkg for pkg in pip_packages if pkg.lower() not in installed]
                if to_install:
                    for pkg in to_install:
                        subprocess.check_call([venv_pip, "install", pkg])
            except subprocess.CalledProcessError as e:
                return False, f"Failed to install packages: {e}"

        if system_deps:
            missing = [dep for dep in system_deps if subprocess.call(["which", dep], stdout=subprocess.PIPE) != 0]
            if missing:
                try:
                    subprocess.check_call(["sudo", "-n", "apt", "install", "-y"] + missing)
                except subprocess.CalledProcessError:
                    sudo_pass = getpass.getpass(f"Enter sudo password to install {', '.join(missing)}: ")
                    try:
                        cmd = ["sudo", "-S", "apt", "install", "-y"] + missing
                        subprocess.check_call(f"echo {sudo_pass} | {' '.join(cmd)}", shell=True)
                    except subprocess.CalledProcessError as e:
                        return False, f"Failed to install system deps: {e}"
        return True, ""

    # Skill Management Methods (Merged from SkillManager)
    def acquire_skill(self, skill_name: str, description: str) -> Skill:
        role_context = f"{skill_name}: {description}" if description else skill_name
        skills = self.auto_acquire_skills(role_context, batch_size=1)
        primary_skill = next((s for s in skills if s.name.lower() == skill_name.lower()), skills[0])
        return primary_skill

    def auto_acquire_skills(self, role_or_context: str, batch_size: int = 10) -> List[Skill]:
        role_info = self.inference_engine.get_role_info(role_or_context)
        all_skills_data = role_info["skills"]
        all_skills = self._fetch_all_skills(role_or_context, all_skills_data)
        
        stored_skills = []
        for skill_data in all_skills:
            skill = Skill(
                agent_id=self.memory_manager.current_agent.id,
                name=skill_data["name"],
                description=skill_data["description"],
                instructions={"type": "text", "content": skill_data["description"]},
                acquired=False
            )
            skill_id = self.memory_manager.memory.store_skill(skill)
            skill.id = skill_id
            stored_skills.append(skill)

        unacquired = [s for s in stored_skills if not s.acquired][:batch_size]
        for skill, skill_data in zip(unacquired, all_skills[:len(unacquired)]):
            code = skill_data.get("code")
            if code and self._is_valid_python(code):
                script_response = self.generate_script(code, skill_name=skill.name)
                file_name = script_response.split("File: ")[1].strip()
                skill.instructions = {"type": "script", "path": f"scripts/{file_name}", "venv": "venvs/auto_ninja_venv"}
                skill.script_id = self.memory_manager.memory.sqlite_store.cursor.lastrowid
            else:
                self._process_skill(skill)
            skill.acquired = True
            self.memory_manager.memory.store_skill(skill)
            summary = f"Skill: {skill.name} - {skill.description or 'No description'}"
            embedding = self.memory_manager.memory.qdrant_store.generate_embedding(summary)
            self.memory_manager.memory.qdrant_store.store_summary(skill, summary, embedding, "skill_summaries")
        return stored_skills

    def _fetch_all_skills(self, role_or_context: str, initial_skills: List[Dict]) -> List[Dict]:
        all_skills = initial_skills.copy()
        for skill in initial_skills:
            sub_prompt = (
                f"List skills required for '{skill['name']}' in a '{role_or_context}' role. "
                "Return JSON: {{'skills': [{{'name': 'str', 'description': 'str', 'code': 'str or null'}}]}}."
            )
            sub_response = self.inference_engine.process(sub_prompt)
            try:
                sub_skills_data = robust_parse_json(sub_response)
                sub_skills = sub_skills_data["skills"]
                for sub_skill in sub_skills:
                    if sub_skill.get("code") and not self._is_valid_python(sub_skill["code"]):
                        sub_skill["code"] = None
                    all_skills.append(sub_skill)
            except ValueError:
                continue
        return all_skills

    def _process_skill(self, skill: Skill):
        prompt = (
            f"Does skill '{skill.name}' (Description: '{skill.description}') need a script for '{self.memory_manager.current_agent.role}'? "
            "Return JSON: {{'needs_script': bool, 'description': str, 'code': str or null}}"
        )
        response = self.inference_engine.process(prompt)
        try:
            details = robust_parse_json(response)
            if details["needs_script"] and details["code"] and self._is_valid_python(details["code"]):
                script_response = self.generate_script(details["code"], skill_name=skill.name)
                file_name = script_response.split("File: ")[1].strip()
                skill.instructions = {"type": "script", "path": f"scripts/{file_name}", "venv": "venvs/auto_ninja_venv"}
                skill.script_id = self.memory_manager.memory.sqlite_store.cursor.lastrowid
            else:
                skill.instructions = {"type": "text", "content": details["description"]}
        except ValueError:
            skill.instructions = {"type": "text", "content": skill.description or "No additional details"}

    def _is_valid_python(self, code: str) -> bool:
        return code and code.lower() != "null" and any(keyword in code for keyword in ["import", "def", "class", "print", "if"])

    def update_skill(self, skill_id: int, description: str) -> Skill:
        skill = self.memory_manager.memory.get_skill(skill_id)
        if not skill or skill.instructions["type"] != "script":
            raise ValueError(f"Skill {skill_id} not found or not script-based")
        file_name = os.path.basename(skill.instructions["path"])
        prompt = f"Update script for '{skill.name}' with description '{description}'. Use sys.argv[1] for parameters."
        self.update_script(prompt, file_name)
        skill.description = description
        self.memory_manager.memory.store_skill(skill)
        return skill

    def delete_skill(self, skill_id: int) -> str:
        skill = self.memory_manager.memory.get_skill(skill_id)
        if not skill or skill.instructions["type"] != "script":
            raise ValueError(f"Skill {skill_id} not found or not script-based")
        file_name = os.path.basename(skill.instructions["path"])
        self.delete_script(file_name)
        self.memory_manager.memory.sqlite_store.cursor.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        self.memory_manager.memory.sqlite_store.conn.commit()
        return f"Skill '{skill.name}' deleted successfully"

    def use_skill(self, skill_id: int, task: str) -> str:
        skill = self.memory_manager.memory.get_skill(skill_id)
        if not skill:
            raise ValueError(f"Skill {skill_id} not found")
        if skill.instructions["type"] == "script":
            file_name = os.path.basename(skill.instructions["path"])
            return self.execute_script(f"{file_name} {task}")
        prompt = f"Using skill '{skill.name}' with instructions '{skill.instructions['content']}', complete: {task}"
        return self.inference_engine.process(prompt)

    def list_skills(self) -> List[Skill]:
        return self.memory_manager.memory.sqlite_store.get_skills_by_agent(self.memory_manager.current_agent.id)

==================================================
File: transformers/data.py
==================================================

from ..config.settings import settings

class DataTransformer:
    def preprocess(self, message: str) -> str:
        # Basic validation
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message must be a non-empty string")
        return message.strip()
    
    def postprocess(self, text: str) -> str:
        cleaned_text = " ".join(text.split())
        return cleaned_text[:settings.MAX_SEQUENCE_LENGTH * 2]

==================================================
File: transformers/__init__.py
==================================================



==================================================
File: models/payload.py
==================================================

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

==================================================
File: models/tool.py
==================================================

# app/models/tool.py
from datetime import datetime
import json

class Tool:
    def __init__(self, id: int = None, agent_id: int = None, name: str = "", description: str = None, instructions: dict = None, usage_frequency: int = None, venv_path: str = None, dependencies: list = None, created_at: str = None, updated_at: str = None, metadata: dict = None):
        self.id = id
        self.agent_id = agent_id  # Added to link tool to an agent
        self.name = name
        self.description = description
        self.instructions = instructions  # e.g., {"script": "print('Hello')", "path": "scripts/x.py"}
        self.usage_frequency = usage_frequency
        self.venv_path = venv_path  # e.g., "venvs/tool_x"
        self.dependencies = dependencies if dependencies else []  # e.g., ["requests", "numpy"]
        self.created_at = created_at if created_at else datetime.now().isoformat()
        self.updated_at = updated_at
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,  # Added
            "name": self.name,
            "description": self.description,
            "instructions": json.dumps(self.instructions) if self.instructions else None,
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
            agent_id=data.get("agent_id"),  # Added
            name=data["name"],
            description=data["description"],
            instructions=json.loads(data["instructions"]) if data["instructions"] else None,
            usage_frequency=data["usage_frequency"],
            venv_path=data["venv_path"],
            dependencies=json.loads(data["dependencies"]) if data["dependencies"] else None,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=json.loads(data["metadata"])
        )

==================================================
File: models/inference_response.py
==================================================

from pydantic import BaseModel

class InferenceResponse(BaseModel):
    prediction: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Hi there! I'm here to help you with any questions you might have."
            }
        }

==================================================
File: models/skill.py
==================================================

# app/models/skill.py
from datetime import datetime
import json

class Skill:
    def __init__(self, id: int = None, agent_id: int = None, name: str = "", description: str = None, instructions: dict = None, proficiency: float = None, script_id: int = None, acquired: bool = False, created_at: str = None, updated_at: str = None, metadata: dict = None):
        self.id = id
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.instructions = instructions
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
            "instructions": self.instructions,  # Already a dict, no need to serialize here
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
            instructions=data["instructions"],  # Already deserialized by sqlite_store
            proficiency=data["proficiency"],
            script_id=data["script_id"],
            acquired=data.get("acquired", False),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=json.loads(data["metadata"]) if data["metadata"] else {}
        )

==================================================
File: models/__init__.py
==================================================

from .interaction import Interaction
from .agent import Agent
from .skill import Skill
from .tool import Tool
from .payload import GrokPayload, create_grok_payload
from .inference_request import InferenceRequest
from .inference_response import InferenceResponse

==================================================
File: models/inference_request.py
==================================================

from pydantic import BaseModel

class InferenceRequest(BaseModel):
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, how can I assist you today?"
            }
        }

==================================================
File: models/interaction.py
==================================================

from datetime import datetime
import json

class Interaction:
    def __init__(self, id: int = None, agent_id: int = 0, timestamp: str = None, user_prompt: str = "", actual_output: str = "", target_output: str = None, parent_id: int = None, metadata: dict = None):
        self.id = id
        self.agent_id = agent_id
        self.timestamp = timestamp if timestamp else datetime.now().isoformat()
        self.user_prompt = user_prompt
        self.actual_output = actual_output
        self.target_output = target_output
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
            parent_id=data["parent_id"],
            metadata=json.loads(data["metadata"])
        )

==================================================
File: models/agent.py
==================================================

# app/models/agent.py
from datetime import datetime
import json

class Agent:
    def __init__(self, id: int = None, name: str = "", role: str = "", role_info: dict = None, created_at: str = None, updated_at: str = None, metadata: dict = None):
        self.id = id
        self.name = name
        self.role = role
        self.role_info = role_info
        self.created_at = created_at if created_at else datetime.now().isoformat()
        self.updated_at = updated_at
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "role_info": json.dumps(self.role_info) if self.role_info else None,
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
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=json.loads(data["metadata"])
        )

==================================================
File: agent/evolution_manager.py
==================================================

# agent/evolution_manager.py
import os
import glob
import subprocess
import logging
import re
import json

logger = logging.getLogger(__name__)

class EvolutionManager:
    def __init__(self, inference_engine, memory_manager):
        self.inference_engine = inference_engine
        self.memory_manager = memory_manager

    def evolve(self) -> str:
        """Condense codebase, assess with Grok, rewrite, and restart."""
        # Step 1: Condense codebase
        codebase = "# Auto Ninja Condensed Codebase\n\n"
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        exclude_dirs = {'__pycache__', 'scripts', 'speech', 'venvs'}
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file.endswith('.py') and file != 'auto_ninja.py':
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        codebase += f"\n# File: {os.path.relpath(file_path, base_dir)}\n{f.read()}\n"
        escaped_codebase = codebase.replace('\n', '\\n').replace('"', '\\"')
        # Step 2: Use generic prompt with agent details
        agent_name = self.memory_manager.current_agent.name
        agent_role = self.memory_manager.current_agent.role
        prompt = EVOLUTION_PROMPT.format(agent_name=agent_name, agent_role=agent_role, codebase=escaped_codebase)
        
        # Step 3: Send to Grok for assessment
        try:
            response = self.inference_engine.process(prompt)
            logger.debug(f"Grok codebase assessment: {response}")
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            result = json.loads(json_match.group(0) if json_match else response)
            assessment = result.get("assessment", "No assessment provided")
            improvements = result.get("improvements", "No improvements suggested")
            revised_code = result.get("revised_code", codebase)
            
            # Step 4: Rewrite codebase
            new_file_path = os.path.join(base_dir, 'auto_ninja_evolved.py')
            with open(new_file_path, 'w') as f:
                f.write(revised_code)
            logger.info(f"Codebase rewritten to {new_file_path}")
            logger.info(f"Assessment: {assessment}\nImprovements: {improvements}")
            
            # Step 5: Generate and execute restart script
            restart_script = os.path.join(base_dir, 'restart.sh')
            with open(restart_script, 'w') as f:
                f.write(f"""#!/bin/bash
                echo "Shutting down current Auto Ninja..."
                pkill -f "python auto_ninja.py"
                sleep 2
                echo "Replacing old codebase..."
                mv {new_file_path} {os.path.join(base_dir, 'auto_ninja.py')}
                echo "Restarting Auto Ninja..."
                python {os.path.join(base_dir, 'auto_ninja.py')} &
                """)
            os.chmod(restart_script, 0o755)
            subprocess.run(['bash', restart_script], check=True)
            logger.info("Restart script executed; Auto Ninja is evolving...")
            return "Evolution complete; restarting with new codebase."
        except Exception as e:
            logger.error(f"Failed to evolve: {str(e)}")
            return f"Evolution failed: {str(e)}"

EVOLUTION_PROMPT = """
I am {agent_name}, a {agent_role} AI agent. Here iis my condensed codebase:
- Files: shared.py (agent initialization), skill_manager.py (skill acquisition), evolution_manager.py (self-evolution), auto_ninja.py (CLI interface), and others in app/ subdirectories.
{codebase}

Assess my codebase for improvements as a {agent_role}. Return a JSON response:
{{
    "assessment": "Brief evaluation of the codebase",
    "improvements": "Suggested changes or optimizations",
    "revised_code": "Complete revised Python codebase as a single file, implementing the improvements"
}}
Return ONLY the JSON string, with no additional text or markdown.
"""

==================================================
File: auto_ninja.py
==================================================
# auto_ninja.py
import sys
import asyncio
import aiohttp
import logging
import speech_recognition as sr
import pyttsx3
from app.scripting.agent_manager import AgentManager
from app.shared import SingletonInference
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_server_status(session):
    try:
        async with session.get('http://localhost:8080/inference/health') as response:
            status = await response.json()
            logger.info(f"Server responded: {status}")
            return status.get('status') == 'healthy'
    except Exception as e:
        logger.error(f"Failed to check server status: {e}")
        return False

def init_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    return engine

def speak(text, engine):
    engine.say(text)
    engine.runAndWait()

def listen(recognizer):
    with sr.Microphone() as source:
        print("Listening... (say 'text' to type instead)")
        audio = recognizer.listen(source, timeout=5)
        try:
            command = recognizer.recognize_google(audio).strip().lower()
            logger.info(f"Recognized speech: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

async def process_command(inference_engine, agent_manager: AgentManager, agent, command, mode, tts_engine, recognizer):
    command = command.strip().lower()
    parts = command.split(" ", 2)
    # Use the full command for matching multi-word commands
    full_cmd = command.split(" ", 2)[:2]  # Take first two words for commands like "list skills"
    cmd = " ".join(full_cmd) if len(full_cmd) == 2 else parts[0] if parts else ""

    # Handle all predefined commands
    if cmd == "exit":
        return False, mode
    elif cmd == "use speech":
        mode = "speech"
        response = "Switched to speech mode. Speak your command or say 'text' to type."
        print(response)
        if mode == "speech":
            speak(response, tts_engine)
    elif cmd == "use text":
        mode = "text"
        response = "Switched to text mode. Type your command."
        print(response)
        if mode == "speech":
            speak(response, tts_engine)
    elif cmd == "list skills":
        try:
            skills = agent_manager.list_skills()
            logger.info(f"Skills for agent {agent.id}: {[s.id for s in skills]}")
            output = "\n".join([f"ID: {s.id}, Name: {s.name}, Acquired: {s.acquired}, Instructions: {s.instructions}" for s in skills])
            print(output or "No skills found.")
            if mode == "speech":
                speak(output or "No skills found.", tts_engine)
        except Exception as e:
            logger.error(f"Failed to list skills: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts_engine)
    elif cmd == "acquire skill":
        try:
            name_desc = parts[2] if len(parts) > 2 else ""
            name, desc = name_desc.split(",", 1) if "," in name_desc else (name_desc, "")
            skill = agent_manager.acquire_skill(name.strip(), desc.strip())
            response = f"Acquired skill '{skill.name}' with ID {skill.id}"
            print(response)
            if mode == "speech":
                speak(response, tts_engine)
        except Exception as e:
            logger.error(f"Failed to acquire skill: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts_engine)
    elif cmd == "use skill":
        try:
            skill_id_task = parts[2] if len(parts) > 2 else ""
            skill_id, task = skill_id_task.split(",", 1) if "," in skill_id_task else (skill_id_task, "")
            skill_id = int(skill_id.strip())
            task = task.strip()

            skills = agent_manager.list_skills()
            skill = next((s for s in skills if s.id == skill_id), None)
            if not skill:
                response = f"Skill ID {skill_id} not found in agent {agent.id}'s skills"
                print(response)
                if mode == "speech":
                    speak(response, tts_engine)
                return True, mode

            output = agent_manager.use_skill(skill_id, task)
            print(output)
            if mode == "speech":
                speak(output, tts_engine)
        except Exception as e:
            logger.error(f"Failed to use skill: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts_engine)
    elif cmd == "update skill":
        try:
            skill_id_desc = parts[2] if len(parts) > 2 else ""
            skill_id, desc = skill_id_desc.split(",", 1) if "," in skill_id_desc else (skill_id_desc, "")
            skill_id = int(skill_id.strip())
            desc = desc.strip()

            skills = agent_manager.list_skills()
            skill = next((s for s in skills if s.id == skill_id), None)
            if not skill:
                response = f"Skill ID {skill_id} not found in agent {agent.id}'s skills"
                print(response)
                if mode == "speech":
                    speak(response, tts_engine)
                return True, mode

            skill = agent_manager.update_skill(skill_id, desc)
            response = f"Updated skill '{skill.name}' with ID {skill_id}"
            print(response)
            if mode == "speech":
                speak(response, tts_engine)
        except Exception as e:
            logger.error(f"Failed to update skill: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts_engine)
    elif cmd == "delete skill":
        try:
            skill_id_str = parts[2] if len(parts) > 2 else ""
            if not skill_id_str:
                response = "Please provide a skill ID to delete (e.g., 'delete skill 59')"
                print(response)
                if mode == "speech":
                    speak(response, tts_engine)
                return True, mode

            skill_id = int(skill_id_str)
            skills = agent_manager.list_skills()
            skill = next((s for s in skills if s.id == skill_id), None)
            if not skill:
                response = f"Skill ID {skill_id} not found in agent {agent.id}'s skills"
                print(response)
                if mode == "speech":
                    speak(response, tts_engine)
                return True, mode

            response = agent_manager.delete_skill(skill_id)
            print(response)
            if mode == "speech":
                speak(response, tts_engine)
        except Exception as e:
            logger.error(f"Failed to delete skill: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts_engine)
    elif cmd == "acquire next":
        try:
            n = int(parts[2]) if len(parts) > 2 else 1
            response = f"Acquire next {n} not implemented yet"
            print(response)
            if mode == "speech":
                speak(response, tts_engine)
        except Exception as e:
            logger.error(f"Failed to acquire next: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts_engine)
    else:
        # Handle script prompts and unrecognized commands
        try:
            action = agent_manager.is_script_prompt(command)
            if action:
                logger.info(f"Detected script command: {command}")
                if agent_manager.pending_action:
                    response = agent_manager.confirm_action(command)
                else:
                    details = {"prompt": command}
                    if action != "generate":
                        from app.scripting.utils import extract_file_name
                        file_name = extract_file_name(command)
                        if not file_name:
                            response = "Please specify a valid script file (e.g., math_addition_001.py)."
                        else:
                            details["file_name"] = file_name
                            response = agent_manager.request_confirmation(action, details)
                    else:
                        response = agent_manager.request_confirmation(action, details)
            else:
                logger.info(f"Processing general prompt: {command}")
                response = inference_engine.process(command)
            print(response)
            if mode == "speech":
                speak(response, tts_engine)
        except Exception as e:
            logger.error(f"Failed to process command: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts_engine)
    return True, mode

async def main_async():
    async with aiohttp.ClientSession() as session:
        if not await check_server_status(session):
            logger.error("Server is not healthy. Exiting...")
            sys.exit(1)

        inference_engine = SingletonInference.get_instance()
        agent_manager = AgentManager(inference_engine, inference_engine.memory_manager)
        agent = inference_engine.memory_manager.current_agent
        print(f"Using agent {agent.name} (Role: {agent.role}). Type your command (e.g., 'acquire skill', 'list skills', 'exit'): ")
        print("Commands: 'acquire skill <name>, <description>', 'use skill <id>, <task>', 'list skills', 'update skill <id>, <description>', 'delete skill <id>', 'acquire next [n]', 'use speech', 'use text', 'exit'")

        mode = "text"  # Default mode
        tts_engine = init_tts()
        recognizer = sr.Recognizer()

        while True:
            if mode == "text":
                command = input("> ").strip()
            else:  # Speech mode
                command = listen(recognizer)
                if command == "text":
                    command = input("Type your command: ").strip()
                elif not command:
                    continue

            continue_running, mode = await process_command(
                inference_engine, agent_manager, agent, command, mode, tts_engine, recognizer
            )
            if not continue_running:
                break

def main():
    logger.info("Starting interaction. Checking server status...")
    asyncio.run(main_async())

if __name__ == "__main__":
    main()