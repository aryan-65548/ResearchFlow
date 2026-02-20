import chromadb
from chromadb.config import Settings
import os

class VectorStore:
    """
    Manages ChromaDB — stores embeddings and retrieves
    relevant chunks based on similarity search.
    """

    def __init__(self, persist_path: str = "./data/chroma_db"):
        """
        persist_path: where ChromaDB saves its files on disk
        Data persists between runs — you don't re-embed every time
        """
        self.persist_path = persist_path
        os.makedirs(persist_path, exist_ok=True)

        # Create ChromaDB client that saves to disk
        self.client = chromadb.PersistentClient(path=persist_path)
        print(f"ChromaDB initialized at: {persist_path}")

    def create_collection(self, collection_name: str):
        """
        A collection is like a table in a regular database.
        Each research paper can have its own collection,
        or you can put all papers in one collection.
        We use one collection for all papers.
        """
        # get_or_create means it won't crash if collection already exists
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # cosine similarity for search
        )
        print(f"Collection '{collection_name}' ready")
        return collection

    def add_chunks(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict]
    ):
        """
        Stores chunks with their embeddings in ChromaDB.
        All 4 lists must be the same length and in the same order.
        """
        collection = self.create_collection(collection_name)

        # ChromaDB stores: ID + embedding + original text + metadata
        # This lets us retrieve the actual text after finding similar vectors
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"Successfully stored {len(ids)} chunks in ChromaDB")
        return collection

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5
    ) -> list[dict]:
        """
        Given a query embedding, finds the most similar chunks.
        n_results: how many chunks to return (top-N)
        """
        collection = self.create_collection(collection_name)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Reformat results into clean list of dicts
        chunks = []
        for i in range(len(results["documents"][0])):
            chunk = {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                # distance: 0.0 = identical, 2.0 = completely different
                "similarity": round(1 - results["distances"][0][i], 4)
            }
            chunks.append(chunk)

        return chunks

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection already has data."""
        try:
            col = self.client.get_collection(collection_name)
            return col.count() > 0
        except:
            return False

    def get_collection_count(self, collection_name: str) -> int:
        """Returns how many chunks are stored in a collection."""
        try:
            col = self.client.get_collection(collection_name)
            return col.count()
        except:
            return 0

    def delete_collection(self, collection_name: str):
        """Wipe a collection — useful for re-processing a paper."""
        try:
            self.client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted")
        except:
            print(f"Collection '{collection_name}' not found")