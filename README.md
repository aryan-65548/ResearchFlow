# ResearchFlow

**Translate, understand, and discover research papers — powered by Groq.**

ResearchFlow is a research assistant built with RAG (Retrieval Augmented Generation). Upload any research paper, ask questions about it, translate sections into 15 languages, and discover related papers from arXiv — using Groq's fast cloud LLM inference.

---

## Features

- **PDF Upload & Processing** — extract, chunk, and embed any research paper
- **RAG-powered Q&A** — ask questions and get answers grounded in the paper
- **Translation** — translate any section into 15 languages with technical term accuracy
- **Simplify** — convert complex academic text into plain English
- **arXiv Search** — search and import papers directly from arXiv
- **Paper Recommender** — get AI-powered recommendations based on your uploaded papers
- **Fast cloud inference** — powered by Groq's LPU-based API, no local GPU needed

---

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| RAG Orchestration | LangChain |
| LLM | Groq API (llama-3.3-70b-versatile / llama-3.1-8b-instant) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| PDF Parsing | PyMuPDF |
| Paper Discovery | arXiv API |

---

## Getting Started

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com/keys)

### Installation

```bash
# Clone the repo
git clone https://github.com/aryan-65548/ResearchFlow.git
cd ResearchFlow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
cp .env.example .env   # then edit .env and paste your key
```

### Run

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (make sure `.env` is **not** committed — it's in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app, pointing it at `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
4. Deploy. Since the LLM now runs via the Groq API, no GPU or local model server is needed — the free Streamlit Cloud tier is enough.

The same `GROQ_API_KEY` env var works for any other host (Render, Railway, Hugging Face Spaces, Docker, etc.) — just set it as an environment variable / secret on that platform.

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
