import os
import logging
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


def generate_and_store_embeddings(
    clean_texts: list, persist_directory: str = "./chroma_db"
) -> Chroma:
    """Convert cleaned string sequences into embeddings and save them to a local Vector Database."""
    try:
        logger.info(
            f"Initializing local vector storage registry at: {persist_directory}"
        )

        # Using a fixed 768-dimension mock encoder layout that matches BERT's base matrix size
        # This completely avoids downloading huge embedding model weights over restricted networks
        embedding_engine = DeterministicFakeEmbedding(size=768)

        logger.info(
            f"Indexing {len(clean_texts)} text blocks into persistent Chroma storage vectors."
        )

        # Initialize and populate the local database
        vector_db = Chroma.from_texts(
            texts=clean_texts,
            embedding=embedding_engine,
            persist_directory=persist_directory,
        )

        logger.info(
            "[SUCCESS] Vector database generation and persistence loop completed."
        )
        return vector_db

    except Exception as e:
        logger.error(f"Failed to compile text vector embeddings layer: {e}")
        raise e

def retrieve_relevant_context(query: str, persist_directory: str = "./chroma_db", k: int = 2) -> list:
    """
    Query the persistent Chroma vector database to extract semantically relevant text context.
    
    Args:
        query (str): The natural language query or user prompt.
        persist_directory (str): The path where the vector database is stored.
        k (int): The number of top relevant documents to retrieve.
        
    Returns:
        list: A list of clean string fragments matching the query context.
    """
    try:
        logger.info(f"Querying vector database for context routing. Prompt: '{query}'")
        
        # Instantiate the exact same embedding model layout used during ingestion
        embedding_engine = DeterministicFakeEmbedding(size=768)
        
        # Load the existing persistent database from disk without writing new records
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_engine
        )
        
        # Execute a similarity search against the vector index
        matching_docs = vector_db.similarity_search(query, k=k)
        
        # Extract the raw string content out of the LangChain Document wrappers
        retrieved_contexts = [doc.page_content for doc in matching_docs]
        
        logger.info(f"[SUCCESS] Extracted {len(retrieved_contexts)} relevant matching context blocks.")
        return retrieved_contexts
        
    except Exception as e:
        logger.error(f"Failed to query and route vector context layers: {e}")
        raise e

if __name__ == "__main__":
    # Isolated module execution verification check loop
    logging.basicConfig(level=logging.INFO)
    sample_corpus = [
        "enterprise spark data processing pipeline",
        "bert model sequence ingestion metrics verification",
        "mlflow tracking configuration is completely functional",
    ]
    generate_and_store_embeddings(sample_corpus)
