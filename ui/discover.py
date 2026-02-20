import streamlit as st
from core.arxiv_client import ArxivClient


def get_arxiv_client():
    """Get or create ArxivClient from session state."""
    if "arxiv_client" not in st.session_state:
        embedder = st.session_state.get("embedder", None)
        st.session_state.arxiv_client = ArxivClient(embedder=embedder)
    return st.session_state.arxiv_client


def render_paper_card(paper: dict, show_similarity: bool = False):
    """
    Renders a single paper card with metadata and action buttons.
    Used for both search results and recommendations.
    """
    with st.container(border=True):

        # Title + similarity badge
        col_title, col_badge = st.columns([4, 1])
        with col_title:
            st.markdown(f"**{paper['title']}**")
        with col_badge:
            if show_similarity:
                pct = paper.get("similarity_pct", 0)
                if pct >= 80:
                    st.success(f"{pct}%")
                elif pct >= 60:
                    st.warning(f"{pct}%")
                else:
                    st.info(f"{pct}%")

        # Metadata row
        st.caption(
            f"Authors: {paper['authors']}  |  "
            f"Published: {paper['published']}  |  "
            f"Category: {paper['categories']}"
        )

        # Abstract preview
        abstract_preview = paper["abstract"][:300] + "..."
        st.write(abstract_preview)

        # Similarity bar if recommendation
        if show_similarity:
            pct = paper.get("similarity_pct", 0)
            st.progress(pct / 100, text=f"Similarity: {pct}%")

        # Action buttons
        col1, col2 = st.columns(2)

        with col1:
            button_key = f"import_{paper['arxiv_id']}"
            if st.button(
                "Import and Process",
                key=button_key,
                use_container_width=True,
                type="primary"
            ):
                _import_paper(paper)

        with col2:
            st.link_button(
                "View on arXiv",
                url=f"https://arxiv.org/abs/{paper['arxiv_id']}",
                use_container_width=True
            )


def _import_paper(paper: dict):
    """
    Downloads and processes a paper from arXiv into the RAG pipeline.
    """
    from core.pdf_loader import PDFLoader
    from core.chunker import Chunker
    from core.embedder import Embedder
    from core.vector_store import VectorStore

    paper_filename = f"{paper['arxiv_id'].replace('/', '_')}.pdf"

    # Check if already imported
    if paper_filename in st.session_state.get("processed_papers", []):
        st.warning(f"'{paper['title'][:40]}...' is already imported!")
        return

    progress = st.progress(0)
    status = st.status(f"Importing: {paper['title'][:50]}...", expanded=True)

    try:
        # Step 1: Download PDF
        status.write("Downloading PDF from arXiv...")
        progress.progress(15)

        client = get_arxiv_client()
        pdf_path = client.download_pdf(paper, save_dir="./data/uploads")

        status.write("Downloaded successfully")
        progress.progress(30)

        # Step 2: Extract text
        status.write("Extracting text...")
        loader = PDFLoader(pdf_path)
        loader.load().extract_text().clean()
        clean_text = loader.get_text()
        metadata = loader.get_metadata()

        status.write(f"Extracted {len(clean_text):,} characters from {metadata['page_count']} pages")
        progress.progress(50)

        # Step 3: Chunk
        status.write("Chunking text...")
        source_name = paper['arxiv_id'].replace("/", "_").replace(".", "_")
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        chunker.split(clean_text, source_name=source_name)

        status.write(f"Created {chunker.get_chunk_count()} chunks")
        progress.progress(65)

        # Step 4: Embed
        status.write("Generating embeddings...")
        if "embedder" not in st.session_state:
            st.session_state.embedder = Embedder()
        embedder = st.session_state.embedder
        embeddings = embedder.embed_texts(chunker.get_texts_only())

        status.write(f"Generated {len(embeddings)} embeddings")
        progress.progress(80)

        # Step 5: Store
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

        progress.progress(95)

        # Step 6: Save to session state
        if "processed_papers" not in st.session_state:
            st.session_state.processed_papers = []
        st.session_state.processed_papers.append(paper_filename)

        if "papers_metadata" not in st.session_state:
            st.session_state.papers_metadata = {}
        st.session_state.papers_metadata[paper_filename] = {
            "pages": metadata["page_count"],
            "chunks": chunker.get_chunk_count(),
            "characters": len(clean_text),
            "source_name": source_name,
            "title": paper["title"],
            "authors": paper["authors"]
        }

        if "papers_text" not in st.session_state:
            st.session_state.papers_text = {}
        st.session_state.papers_text[paper_filename] = clean_text

        progress.progress(100)
        status.update(label="Import complete!", state="complete")
        loader.close()

        st.success(f"'{paper['title'][:50]}...' imported! Go to Chat or Translate tab.")

    except Exception as e:
        status.update(label="Import failed", state="error")
        st.error(f"Error: {str(e)}")
        progress.progress(0)


def render_search_section():
    """Renders the arXiv search UI."""
    st.subheader("Search arXiv Papers")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search query",
            placeholder="e.g. transformer attention mechanism, BERT language model...",
            label_visibility="collapsed"
        )
    with col2:
        search_btn = st.button("Search", use_container_width=True, type="primary")

    if search_btn and query.strip():
        with st.spinner(f"Searching arXiv for '{query}'..."):
            try:
                client = get_arxiv_client()
                results = client.search(query, max_results=10)
                st.session_state["search_results"] = results
                st.session_state["last_query"] = query
            except Exception as e:
                st.error(f"Search failed: {str(e)}")

    # Display search results
    if "search_results" in st.session_state and st.session_state.search_results:
        results = st.session_state.search_results
        query_label = st.session_state.get("last_query", "")
        st.write(f"**{len(results)} results for '{query_label}':**")
        st.divider()

        for paper in results:
            render_paper_card(paper, show_similarity=False)
            st.write("")


def render_recommendations_section():
    """Renders the paper recommendations UI."""
    st.subheader("Recommended For You")

    if "processed_papers" not in st.session_state or not st.session_state.processed_papers:
        st.info("Upload or import a paper first to get recommendations.")
        return

    papers = st.session_state.processed_papers
    selected_paper = st.selectbox(
        "Base recommendations on:",
        options=papers,
        format_func=lambda x: st.session_state.get("papers_metadata", {}).get(x, {}).get("title", x)
    )

    if st.button("Find Similar Papers", use_container_width=True, type="primary"):
        papers_text = st.session_state.get("papers_text", {})

        if selected_paper not in papers_text:
            st.warning("Paper text not available for recommendations. Try importing via arXiv search.")
            return

        paper_text = papers_text[selected_paper]

        with st.spinner("Finding similar papers on arXiv... (30-60 seconds)"):
            try:
                client = get_arxiv_client()
                keywords = client.extract_keywords(paper_text)
                st.caption(f"Searching with keywords: {keywords[:80]}...")

                recommendations = client.get_recommendations(
                    uploaded_paper_text=paper_text,
                    query_keywords=keywords,
                    top_n=5,
                    candidate_pool=20
                )

                st.session_state["recommendations"] = recommendations
                st.session_state["rec_based_on"] = selected_paper

            except Exception as e:
                st.error(f"Recommendation failed: {str(e)}")
                st.info("Make sure you have internet connection.")

    # Display recommendations
    if "recommendations" in st.session_state and st.session_state.recommendations:
        recs = st.session_state.recommendations
        based_on = st.session_state.get("rec_based_on", "")
        based_on_title = st.session_state.get("papers_metadata", {}).get(based_on, {}).get("title", based_on)

        st.write(f"**Top 5 papers similar to:** {based_on_title[:60]}")
        st.divider()

        for paper in recs:
            render_paper_card(paper, show_similarity=True)
            st.write("")


def render_discover_section():
    """
    Main entry point — renders full Discover section
    inside the Upload tab.
    """
    st.divider()
    st.header("Discover Papers")
    st.write("Search arXiv or get AI-powered recommendations based on your uploaded papers.")

    discover_tab1, discover_tab2 = st.tabs(["Search arXiv", "Recommendations"])

    with discover_tab1:
        render_search_section()

    with discover_tab2:
        render_recommendations_section()