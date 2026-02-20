import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()

        # Model selection
        st.subheader(" Model")
        model = st.selectbox(
            "Ollama Model",
            options=["qwen2.5:7b", "llama3.1:8b", "mistral:7b", "gemma2:9b"],
            index=0
        )

        # Language selection
        st.subheader("🌐 Translation Language")
        language = st.selectbox(
            "Target Language",
            options=[
                "Hindi", "Gujarati", "Spanish", "French", "German",
                "Chinese", "Japanese", "Arabic", "Portuguese", "Italian",
                "Korean", "Russian", "Dutch", "Turkish", "Bengali"
            ],
            index=0
        )

        # Retrieval settings
        st.subheader("RAG Settings")
        n_results = st.slider(
            "Chunks to retrieve",
            min_value=2,
            max_value=10,
            value=5,
            help="More chunks = more context but slower response"
        )

        chunk_size = st.slider(
            "Chunk size",
            min_value=200,
            max_value=1000,
            value=500,
            step=50,
            help="Characters per chunk when processing PDF"
        )

        st.divider()

        # Uploaded papers list
        st.subheader(" Loaded Papers")
        if "processed_papers" in st.session_state and st.session_state.processed_papers:
            for paper in st.session_state.processed_papers:
                st.success(f" {paper}")
        else:
            st.info("No papers loaded yet")

        st.divider()

        # Live stats
        st.subheader(" Stats")
        if "processed_papers" in st.session_state:
            st.metric("Papers Loaded", len(st.session_state.processed_papers))

        if "chat_history" in st.session_state:
            st.metric("Messages Exchanged", len(st.session_state.chat_history))

        if "vector_store" in st.session_state:
            try:
                count = st.session_state.vector_store.get_collection_count("research_papers")
                st.metric("Chunks in DB", count)
            except:
                pass

        st.divider()
        st.caption("Research Paper Translator v1.0")
        st.caption("Powered by Ollama + ChromaDB")

    return {
        "model": model,
        "language": language,
        "n_results": n_results,
        "chunk_size": chunk_size
    }