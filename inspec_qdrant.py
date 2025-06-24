from qdrant_client import QdrantClient
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Path matches what’s used in MemoryManager
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "interaction_summaries"

def inspect_qdrant():
    # Connect to the embedded Qdrant instance
    client = QdrantClient(path=QDRANT_PATH)
    
    try:
        # Check if collection exists
        collection_info = client.get_collection(COLLECTION_NAME)
        logger.info(f"Collection info: {collection_info}")
        
        # Retrieve all points (summaries)
        points = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,  # Omit vectors for readability
            limit=100  # adjust as needed
        )[0]
        
        if not points:
            logger.info("No points found in collection.")
        else:
            for point in points:
                logger.info(f"Point ID: {point.id}, Payload: {point.payload}")
                
    except Exception as e:
        logger.error(f"Error accessing Qdrant: {str(e)}")

if __name__ == "__main__":
    inspect_qdrant()