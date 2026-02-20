from core.embedder import Embedder
from core.vector_store import VectorStore

class Retriever:
    """
    Connects the Embedder and VectorStore to retrieve
    the most relevant chunks for any given query.
    This is the core of the RAG pipeline.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        collection_name: str = "research_papers",
        n_results: int = 5
    ):
        """
        embedder: your Embedder instance (from Day 4)
        vector_store: your VectorStore instance (from Day 4)
        collection_name: which ChromaDB collection to search
        n_results: how many chunks to retrieve per query
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.n_results = n_results

    def retrieve(self, query: str) -> list[dict]:
        """
        Main method — takes a question, returns relevant chunks.

        Returns list of dicts, each containing:
        - text: the chunk content
        - metadata: source, chunk_index, etc.
        - similarity: how relevant (0.0 to 1.0, higher = better)
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Step 1: Convert the question into a vector
        query_embedding = self.embedder.embed_single(query)

        # Step 2: Search ChromaDB for similar vectors
        results = self.vector_store.search(
            collection_name=self.collection_name,
            query_embedding=query_embedding,
            n_results=self.n_results
        )

        return results

    def retrieve_as_context(self, query: str) -> str:
        """
        Same as retrieve() but formats results into a single
        string that can be directly injected into an LLM prompt.

        This is what we pass to Ollama tomorrow as context.
        """
        chunks = self.retrieve(query)

        if not chunks:
            return "No relevant context found."

        # Format each chunk clearly so the LLM understands the structure
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"].get("source", "unknown")
            similarity = chunk["similarity"]
            text = chunk["text"]

            context_parts.append(
                f"[CONTEXT {i+1} | Source: {source} | Relevance: {similarity}]\n{text}"
            )

        # Join all chunks with clear separator
        full_context = "\n\n---\n\n".join(context_parts)
        return full_context

    def retrieve_with_scores(self, query: str) -> tuple[list[dict], float]:
        """
        Returns chunks AND the average similarity score.
        Useful for knowing if the retrieved context is actually relevant.
        If average score is very low (< 0.3), the paper probably
        doesn't contain good info about the query.
        """
        chunks = self.retrieve(query)

        if not chunks:
            return [], 0.0

        avg_score = sum(c["similarity"] for c in chunks) / len(chunks)
        return chunks, round(avg_score, 4)

    def is_relevant(self, query: str, threshold: float = 0.3) -> bool:
        """
        Quick check — does the paper even contain info about this query?
        threshold: minimum average similarity to consider relevant
        """
        _, avg_score = self.retrieve_with_scores(query)
        return avg_score >= threshold

    def set_n_results(self, n: int):
        """Change how many chunks to retrieve."""
        self.n_results = n

    def set_collection(self, collection_name: str):
        """Switch to a different collection (different paper set)."""
        self.collection_name = collection_name