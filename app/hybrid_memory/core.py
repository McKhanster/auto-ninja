# app/hybrid_memory/core.py
from typing import List
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

    def get_tools_for_agent(self, agent: Agent) -> List[Tool]:
        """Retrieve all tools associated with an agent."""
        return self.sqlite_store.get_tools_for_agent(agent.id)