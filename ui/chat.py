import streamlit as st
from core.embedder import Embedder
from core.vector_store import VectorStore
from core.retriever import Retriever
from core.translator import Translator


def get_pipeline(settings: dict):
    """
    Initializes or retrieves the RAG pipeline from session state.
    We cache it so it doesn't reload the model on every message.
    """
    # Check if pipeline already exists in session state
    pipeline_key = f"pipeline_{settings['model']}"

    if pipeline_key not in st.session_state:
        with st.spinner("Loading AI pipeline..."):
            # Reuse embedder and vector store if already created during upload
            if "embedder" not in st.session_state:
                st.session_state.embedder = Embedder()

            if "vector_store" not in st.session_state:
                st.session_state.vector_store = VectorStore()

            embedder = st.session_state.embedder
            store = st.session_state.vector_store

            retriever = Retriever(
                embedder=embedder,
                vector_store=store,
                collection_name="research_papers",
                n_results=settings["n_results"]
            )

            translator = Translator(
                retriever=retriever,
                model_name=settings["model"]
            )

            st.session_state[pipeline_key] = {
                "retriever": retriever,
                "translator": translator
            }

    return st.session_state[pipeline_key]


def render_chat_page(settings: dict):
    """
    Renders the full chat interface.
    """
    st.header("💬 Ask Questions About The Paper")

    # Guard: check if any paper has been processed
    if "processed_papers" not in st.session_state or not st.session_state.processed_papers:
        st.warning("⚠️ No paper loaded yet. Please upload a paper first in the Upload tab.")
        return

    # Show which papers are loaded
    with st.expander(f"📚 {len(st.session_state.processed_papers)} paper(s) loaded"):
        for paper in st.session_state.processed_papers:
            meta = st.session_state.get("papers_metadata", {}).get(paper, {})
            st.write(f"✅ **{paper}** — {meta.get('pages', '?')} pages, {meta.get('chunks', '?')} chunks")

    st.divider()

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Show context expander for assistant messages
            if message["role"] == "assistant" and "context" in message:
                with st.expander("🔍 View retrieved context"):
                    for i, chunk in enumerate(message["context"]):
                        st.caption(f"Chunk {i+1} | Relevance: {chunk['similarity']}")
                        st.text(chunk["text"][:300])
                        st.divider()

    # Chat input at bottom
    question = st.chat_input("Ask anything about the paper...")

    if question:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(question)

        # Add to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    pipeline = get_pipeline(settings)
                    translator = pipeline["translator"]

                    result = translator.answer_question(question)
                    answer = result["answer"]
                    context = result["context_used"]
                    relevance = result["avg_relevance"]

                    # Display answer
                    st.write(answer)

                    # Show relevance score
                    if relevance >= 0.25:
                        st.caption(f"📊 Relevance score: {relevance}")
                    else:
                        st.caption("⚠️ Low relevance — the paper may not cover this topic")

                    # Show retrieved context in expander
                    if context:
                        with st.expander("🔍 View retrieved context"):
                            for i, chunk in enumerate(context):
                                st.caption(f"Chunk {i+1} | Relevance: {chunk['similarity']}")
                                st.text(chunk["text"][:300])
                                st.divider()

                    # Save assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "context": context
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}\n\nMake sure Ollama is running: `ollama serve`"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ {error_msg}"
                    })

    # Clear chat button
    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()