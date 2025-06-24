# app/hybrid_memory/sqlite_store.py
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

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id INTEGER,
                timestamp TEXT,
                user_prompt TEXT,
                actual_output TEXT,
                target_output TEXT,
                intent TEXT,
                parent_id INTEGER,
                metadata TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(id),
                FOREIGN KEY (parent_id) REFERENCES interactions(id)
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS action_intents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT NOT NULL,
                embedding BLOB NOT NULL,
                parameters TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
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
                parameters TEXT,
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

        self.cursor.execute('DROP TABLE IF EXISTS agent_skills')
        self.conn.commit()

    def store_interaction(self, interaction: Interaction) -> int:
        data = interaction.to_dict()
        self.cursor.execute('''
            INSERT INTO interactions (agent_id, timestamp, user_prompt, actual_output, target_output, intent, parent_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data["agent_id"],
            data["timestamp"],
            data["user_prompt"],
            data["actual_output"],
            data["target_output"],
            data["intent"],
            data["parent_id"],
            data["metadata"]
        ))
        interaction_id = self.cursor.lastrowid
        self.conn.commit()
        interaction.id = interaction_id
        return interaction_id

    def get_interaction(self, interaction_id: int) -> Interaction:
        self.cursor.execute('''
            SELECT id, agent_id, timestamp, user_prompt, actual_output, target_output, intent, parent_id, metadata
            FROM interactions WHERE id = ?
        ''', (interaction_id,))
        data = self.cursor.fetchone()
        if data:
            return Interaction.from_dict({
                "id": data[0],
                "agent_id": data[1],
                "timestamp": data[2],
                "user_prompt": data[3],
                "actual_output": data[4],
                "target_output": data[5],
                "intent": data[6],
                "parent_id": data[7],
                "metadata": data[8]
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
            "intent": row[6],
            "parent_id": row[7],
            "metadata": row[8]
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

        if hasattr(agent, 'tool_ids') and agent.tool_ids:
            tool_ids = [int(tid) for tid in agent.tool_ids.split(",") if tid]
            for tool_id in tool_ids:
                self.cursor.execute('INSERT OR IGNORE INTO agent_tools (agent_id, tool_id) VALUES (?, ?)', (agent_id, tool_id))

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
            self.cursor.execute('SELECT tool_id FROM agent_tools WHERE agent_id = ?', (agent_id,))
            tool_ids = [row[0] for row in self.cursor.fetchall()]
            agent.tool_ids = ",".join(str(tid) for tid in tool_ids) if tool_ids else ""
            return agent
        return None

    def store_skill(self, skill: Skill) -> int:
        data = skill.to_dict()
        instructions_json = json.dumps(data["instructions"]) if data["instructions"] is not None else None
        self.cursor.execute('''
            INSERT INTO skills (agent_id, name, description, instructions, proficiency, script_id, acquired, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data["agent_id"],
            data["name"],
            data["description"],
            instructions_json,
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
            INSERT INTO tools (agent_id, name, description, instructions, parameters, usage_frequency, venv_path, dependencies, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data["agent_id"],
            data["name"],
            data["description"],
            data["instructions"],
            data["parameters"],
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
                "parameters": data[5],
                "usage_frequency": data[6],
                "venv_path": data[7],
                "dependencies": data[8],
                "created_at": data[9],
                "updated_at": data[10],
                "metadata": data[11]
            })
        return None
    
    def get_skills_by_agent(self, agent_id: int) -> List[Skill]:
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

    def get_tools_for_agent(self, agent_id: int) -> List[Tool]:
        """Retrieve all tools associated with an agent_id."""
        query = '''
            SELECT t.* FROM tools t
            JOIN agent_tools at ON t.id = at.tool_id
            WHERE at.agent_id = ?
        '''
        self.cursor.execute(query, (agent_id,))
        rows = self.cursor.fetchall()
        tools = []
        for row in rows:
            tool = Tool(
                id=row[0],
                agent_id=row[1],
                name=row[2],
                description=row[3],
                instructions=json.loads(row[4]) if row[4] else None,
                parameters=json.loads(row[5]) if row[5] else None,
                usage_frequency=row[6],
                venv_path=row[7],
                dependencies=json.loads(row[8]) if row[8] else None,
                created_at=row[9],
                updated_at=row[10],
                metadata=json.loads(row[11]) if row[11] else {}
            )
            tools.append(tool)
        return tools