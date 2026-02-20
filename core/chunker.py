from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

class Chunker:
    """
    Splits extracted PDF text into overlapping chunks for RAG.
    """

    def __init__(
        self,
        chunk_size: int = 500,       # max characters per chunk
        chunk_overlap: int = 50,     # characters repeated between chunks
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []             # will hold final chunk dictionaries

        # RecursiveCharacterTextSplitter tries to split on:
        # 1. Double newlines (paragraphs) first
        # 2. Single newlines second
        # 3. Sentences (. ! ?) third
        # 4. Words last
        # This keeps chunks semantically meaningful
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " "],
            length_function=len
        )

    def split(self, text: str, source_name: str = "unknown"):
        """
        Takes clean text and splits it into chunks.
        source_name: the PDF filename, stored with each chunk for reference.
        """
        if not text or not text.strip():
            raise ValueError("Text is empty — nothing to chunk")

        # Split the text
        raw_chunks = self.splitter.split_text(text)

        # Wrap each chunk in a dictionary with metadata
        # This metadata travels with the chunk into ChromaDB
        self.chunks = []
        for index, chunk_text in enumerate(raw_chunks):
            chunk = {
                "id": f"{source_name}_chunk_{index}",   # unique ID
                "text": chunk_text,                      # the actual content
                "metadata": {
                    "source": source_name,               # which paper
                    "chunk_index": index,                # position in paper
                    "total_chunks": len(raw_chunks),     # total chunks
                    "char_count": len(chunk_text)        # size of this chunk
                }
            }
            self.chunks.append(chunk)

        return self

    def get_chunks(self):
        """Returns list of chunk dictionaries."""
        return self.chunks

    def get_chunk_count(self):
        """Returns how many chunks were created."""
        return len(self.chunks)

    def get_texts_only(self):
        """Returns just the text strings — needed for embedding later."""
        return [chunk["text"] for chunk in self.chunks]

    def get_metadatas_only(self):
        """Returns just the metadata dicts — needed for ChromaDB later."""
        return [chunk["metadata"] for chunk in self.chunks]

    def get_ids_only(self):
        """Returns just the IDs — needed for ChromaDB later."""
        return [chunk["id"] for chunk in self.chunks]

    def preview(self, num_chunks: int = 3):
        """
        Prints first N chunks so you can visually inspect quality.
        Useful for debugging.
        """
        print(f"\nTotal chunks created: {self.get_chunk_count()}")
        print(f"Chunk size: {self.chunk_size} | Overlap: {self.chunk_overlap}\n")

        for i, chunk in enumerate(self.chunks[:num_chunks]):
            print(f"--- CHUNK {i} ---")
            print(f"ID: {chunk['id']}")
            print(f"Characters: {chunk['metadata']['char_count']}")
            print(f"Text preview: {chunk['text'][:200]}...")
            print()