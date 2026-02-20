from sentence_transformers import SentenceTransformer
import os

class Embedder:
    """
    Converts text chunks into numerical vectors using
    a local sentence transformer model.
    Runs 100% locally — no API needed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        model_name: the embedding model to use
        all-MiniLM-L6-v2 is small, fast, and good quality
        Downloads automatically on first run (~80MB)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print("Embedding model loaded!")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Takes a list of text strings.
        Returns a list of vectors (each vector is a list of floats).
        """
        if not texts:
            raise ValueError("No texts provided to embed")

        print(f"Embedding {len(texts)} chunks...")

        # encode() does the heavy lifting
        # convert_to_tensor=False gives us plain Python lists
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=True    # shows a progress bar in terminal
        )

        # Convert from numpy array to plain Python list
        # ChromaDB needs plain Python lists, not numpy arrays
        embeddings_list = [embedding.tolist() for embedding in embeddings]

        print(f"Created {len(embeddings_list)} embeddings")
        print(f"Each embedding has {len(embeddings_list[0])} dimensions")

        return embeddings_list

    def embed_single(self, text: str) -> list[float]:
        """
        Embeds a single piece of text.
        Used later when embedding the user's question for search.
        """
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()

    def get_model_name(self):
        return self.model_name