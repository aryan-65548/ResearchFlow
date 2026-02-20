import streamlit as st
import os
import tempfile
from core.pdf_loader import PDFLoader
from core.chunker import Chunker
from core.embedder import Embedder
from core.vector_store import VectorStore
from ui.discover import render_discover_section


def render_upload_page(settings: dict):
    st.header(" Upload Research Paper")
    st.write("Upload a PDF research paper to begin translating and asking questions.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any research paper in PDF format"
    )

    if uploaded_file is not None:
        file_size_kb = round(len(uploaded_file.getvalue()) / 1024, 2)
        col1, col2 = st.columns(2)
        col1.metric("File Name", uploaded_file.name)
        col2.metric("File Size", f"{file_size_kb} KB")

        st.divider()

        if st.button("🚀 Process Paper", type="primary", use_container_width=True):
            if uploaded_file.name in st.session_state.get("processed_papers", []):
                st.warning(f" '{uploaded_file.name}' is already processed! Upload a different paper or clear the database.")
            else:
                _process_pdf(uploaded_file, settings)

    else:
        st.info("👆 Upload a PDF above to get started")

        with st.expander(" How it works"):
            st.write("""
            1. **Upload** your research paper PDF
            2. **Processing** — we extract, chunk, and embed the text
            3. **Ask questions** about the paper in natural language
            4. **Translate** any section into your chosen language
            5. **Simplify** complex academic text into plain English
            """)

    # Database management section
    st.divider()
    st.subheader(" Database Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(" Clear All Papers", use_container_width=True):
            try:
                if "vector_store" in st.session_state:
                    st.session_state.vector_store.delete_collection("research_papers")
                st.session_state.processed_papers = []
                st.session_state.chat_history = []
                st.session_state.papers_text = {}
                keys_to_delete = [k for k in st.session_state if k.startswith("pipeline_")]
                for key in keys_to_delete:
                    del st.session_state[key]
                st.success(" Database cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing: {str(e)}")

    with col2:
        count = 0
        if "vector_store" in st.session_state:
            try:
                count = st.session_state.vector_store.get_collection_count("research_papers")
            except:
                pass
        st.metric("Chunks in database", count)

    # Discover section (arXiv search + recommendations)
    render_discover_section()


def _process_pdf(uploaded_file, settings: dict):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    source_name = os.path.splitext(uploaded_file.name)[0]
    source_name = source_name.replace(" ", "_").lower()

    progress = st.progress(0)
    status = st.status("Starting processing...", expanded=True)

    try:
        # Step 1: Extract text
        status.write(" Extracting text from PDF...")
        progress.progress(10)

        loader = PDFLoader(tmp_path)
        loader.load().extract_text().clean()
        clean_text = loader.get_text()
        metadata = loader.get_metadata()

        status.write(f" Extracted {len(clean_text):,} characters from {metadata['page_count']} pages")
        progress.progress(30)

        # Step 2: Chunk
        status.write(" Splitting into chunks...")
        chunker = Chunker(chunk_size=settings["chunk_size"], chunk_overlap=50)
        chunker.split(clean_text, source_name=source_name)
        chunk_count = chunker.get_chunk_count()

        status.write(f" Created {chunk_count} chunks")
        progress.progress(50)

        # Step 3: Embed
        status.write(" Generating embeddings (this takes ~30 seconds)...")

        if "embedder" not in st.session_state:
            st.session_state.embedder = Embedder()

        embedder = st.session_state.embedder
        embeddings = embedder.embed_texts(chunker.get_texts_only())

        status.write(f" Generated {len(embeddings)} embeddings")
        progress.progress(75)

        # Step 4: Store in ChromaDB
        status.write("Storing in vector database...")

        if "vector_store" not in st.session_state:
            st.session_state.vector_store = VectorStore()

        store = st.session_state.vector_store
        store.add_chunks(
            collection_name="research_papers",
            ids=chunker.get_ids_only(),
            embeddings=embeddings,
            texts=chunker.get_texts_only(),
            metadatas=chunker.get_metadatas_only()
        )

        status.write(f"Stored {chunk_count} chunks in ChromaDB")
        progress.progress(95)

        # Step 5: Save to session state
        if "processed_papers" not in st.session_state:
            st.session_state.processed_papers = []

        if uploaded_file.name not in st.session_state.processed_papers:
            st.session_state.processed_papers.append(uploaded_file.name)

        if "papers_metadata" not in st.session_state:
            st.session_state.papers_metadata = {}

        st.session_state.papers_metadata[uploaded_file.name] = {
            "pages": metadata["page_count"],
            "chunks": chunk_count,
            "characters": len(clean_text),
            "source_name": source_name
        }

        # Store clean text for recommender
        if "papers_text" not in st.session_state:
            st.session_state.papers_text = {}
        st.session_state.papers_text[uploaded_file.name] = clean_text

        progress.progress(100)
        status.update(label=" Processing complete!", state="complete")

        loader.close()
        os.unlink(tmp_path)

        st.success("🎉 Paper processed successfully! Go to the Chat or Translate tab.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Pages", metadata["page_count"])
        col2.metric("Chunks Created", chunk_count)
        col3.metric("Characters", f"{len(clean_text):,}")

    except Exception as e:
        progress.progress(0)
        status.update(label=" Processing failed", state="error")
        st.error(f"Error: {str(e)}")
        st.info("Make sure your PDF is not password protected and try again.")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)