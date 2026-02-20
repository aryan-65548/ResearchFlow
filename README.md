# Lexara

**Translate, understand, and discover research papers — locally.**

Lexara is a fully local, privacy-first research assistant built with RAG (Retrieval Augmented Generation). Upload any research paper, ask questions about it, translate sections into 15 languages, and discover related papers from arXiv — all running on your machine with no API costs.

---

## Features

- **PDF Upload & Processing** — extract, chunk, and embed any research paper
- **RAG-powered Q&A** — ask questions and get answers grounded in the paper
- **Translation** — translate any section into 15 languages with technical term accuracy
- **Simplify** — convert complex academic text into plain English
- **arXiv Search** — search and import papers directly from arXiv
- **Paper Recommender** — get AI-powered recommendations based on your uploaded papers
- **100% Local** — no API keys, no data leaves your machine

---

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| RAG Orchestration | LangChain |
| LLM | Ollama (qwen2.5:7b / llama3.2:3b) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| PDF Parsing | PyMuPDF |
| Paper Discovery | arXiv API |

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/lexara.git
cd lexara

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama model
ollama pull llama3.2:3b
```

### Run

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — start Lexara
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. **Upload** a research paper PDF in the Upload tab
2. **Chat** — ask any question about the paper in the Chat tab
3. **Translate** — paste any section and translate or simplify it
4. **Discover** — search arXiv or get recommendations based on your paper

---

## Project Structure

```
lexara/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── core/
│   ├── pdf_loader.py       # PDF text extraction
│   ├── chunker.py          # Text splitting
│   ├── embedder.py         # Vector embeddings
│   ├── vector_store.py     # ChromaDB storage
│   ├── retriever.py        # Similarity search
│   ├── translator.py       # LLM translation + QA
│   └── arxiv_client.py     # arXiv search + recommender
├── ui/
│   ├── sidebar.py
│   ├── upload.py
│   ├── chat.py
│   ├── translate.py
│   └── discover.py
└── data/
    ├── uploads/
    └── chroma_db/
```

---

## Supported Languages

Hindi, Gujarati, Spanish, French, German, Chinese, Japanese, Arabic, Portuguese, Italian, Korean, Russian, Dutch, Turkish, Bengali

---

## License

MIT
