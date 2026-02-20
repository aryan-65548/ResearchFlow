import fitz  # this is PyMuPDF, imported as fitz (its original name)
import os
import re

class PDFLoader:
    """
    Handles everything related to loading and extracting text from PDF files.
    """

    def __init__(self, file_path: str):
        """
        file_path: the path to the PDF file on your computer
        """
        self.file_path = file_path
        self.document = None      # will hold the opened PDF
        self.raw_text = ""        # raw extracted text, page by page
        self.clean_text = ""      # cleaned version ready for chunking
        self.metadata = {}        # title, author, page count etc.

    def load(self):
        """
        Opens the PDF file using PyMuPDF.
        Call this first before anything else.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"PDF not found at: {self.file_path}")

        self.document = fitz.open(self.file_path)
        self._extract_metadata()
        return self  # allows chaining: PDFLoader(path).load().extract()

    def _extract_metadata(self):
        """
        Pulls basic info about the PDF itself.
        Private method (underscore prefix = internal use only).
        """
        meta = self.document.metadata
        self.metadata = {
            "title": meta.get("title", "Unknown Title"),
            "author": meta.get("author", "Unknown Author"),
            "page_count": len(self.document),
            "file_name": os.path.basename(self.file_path),
            "file_size_kb": round(os.path.getsize(self.file_path) / 1024, 2)
        }

    def extract_text(self):
        """
        Goes through every page and extracts raw text.
        Joins pages with a special marker so we know where pages begin.
        """
        if self.document is None:
            raise RuntimeError("Call load() before extract_text()")

        pages_text = []

        for page_number in range(len(self.document)):
            page = self.document[page_number]

            # extract text — "text" mode gives plain text
            # "dict" mode would give positions too (we don't need that now)
            text = page.get_text("text")

            if text.strip():  # skip completely empty pages
                pages_text.append(f"\n--- PAGE {page_number + 1} ---\n{text}")

        self.raw_text = "\n".join(pages_text)
        return self

    def clean(self):
        """
        Cleans the raw extracted text.
        Research PDFs have a lot of noise we need to remove.
        """
        if not self.raw_text:
            raise RuntimeError("Call extract_text() before clean()")

        text = self.raw_text

        # Step 1: Remove page markers we added (optional, keep for debugging)
        # text = re.sub(r'\n--- PAGE \d+ ---\n', '\n', text)

        # Step 2: Remove excessive whitespace
        # \s matches any whitespace, {3,} means 3 or more times
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Step 3: Remove hyphenation at line breaks
        # Research papers split words like "meth-\nod" across lines
        text = re.sub(r'-\n', '', text)

        # Step 4: Join lines that are clearly part of same paragraph
        # A line ending without punctuation that continues lowercase = same paragraph
        text = re.sub(r'(?<![.!?])\n(?=[a-z])', ' ', text)

        # Step 5: Remove weird unicode characters that PDFs sometimes have
        text = text.encode('utf-8', errors='ignore').decode('utf-8')

        # Step 6: Strip leading/trailing whitespace
        text = text.strip()

        self.clean_text = text
        return self

    def get_text(self):
        """Returns the final clean text."""
        return self.clean_text

    def get_metadata(self):
        """Returns PDF metadata dictionary."""
        return self.metadata

    def get_page_count(self):
        """Returns number of pages."""
        return self.metadata.get("page_count", 0)

    def close(self):
        """Always close the document when done to free memory."""
        if self.document:
            self.document.close()