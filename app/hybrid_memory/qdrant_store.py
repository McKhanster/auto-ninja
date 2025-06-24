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