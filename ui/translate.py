import streamlit as st
from core.embedder import Embedder
from core.vector_store import VectorStore
from core.retriever import Retriever
from core.translator import Translator


def get_pipeline(settings: dict):
    pipeline_key = f"pipeline_{settings['model']}"

    if pipeline_key not in st.session_state:
        with st.spinner("Loading AI pipeline..."):
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


def render_translate_page(settings: dict):
    st.header("🌐 Translate Research Paper Text")

    if "processed_papers" not in st.session_state or not st.session_state.processed_papers:
        st.warning(" No paper loaded yet. Please upload a paper first in the Upload tab.")
        return

    st.write("Paste any text from the paper below to translate or simplify it.")
    st.divider()

    # Input section
    col1, col2 = st.columns([3, 1])

    with col1:
        prefill = st.session_state.pop("prefill_text", "")
        input_text = st.text_area(
            " Text to translate",
            value=prefill,
            height=200,
            placeholder="Paste any section from the research paper here...",
            help="You can paste directly from the paper or type any text"
        )

    with col2:
        st.write("**Options**")
        language = settings["language"]
        st.info(f"🌐 Target:\n**{language}**")
        st.caption("Change language in sidebar ⚙️")

        use_context = st.toggle(
            "Use RAG context",
            value=True,
            help="Uses retrieved paper chunks to improve technical term translation"
        )

        if input_text:
            st.caption(f" {len(input_text)} characters")

    st.divider()

    # Action buttons
    col_translate, col_simplify, col_clear = st.columns([2, 2, 1])

    with col_translate:
        translate_btn = st.button(
            f" Translate to {language}",
            type="primary",
            use_container_width=True,
            disabled=not input_text.strip()
        )

    with col_simplify:
        simplify_btn = st.button(
            " Simplify to Plain English",
            use_container_width=True,
            disabled=not input_text.strip()
        )

    with col_clear:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    # Translation result
    if translate_btn and input_text.strip():
        st.divider()
        with st.spinner(f"Translating to {language}... (10-30 seconds)"):
            try:
                pipeline = get_pipeline(settings)
                translator = pipeline["translator"]

                result = translator.translate(
                    text=input_text,
                    target_language=language,
                    use_context=use_context
                )

                translation = result["translation"]
                context_used = result["context_used"]

                st.subheader(f" {language} Translation")
                st.text_area(
                    "Translation output",
                    value=translation,
                    height=200,
                    label_visibility="collapsed"
                )

                st.download_button(
                    label=" Download Translation",
                    data=f"ORIGINAL:\n{input_text}\n\n{language.upper()} TRANSLATION:\n{translation}",
                    file_name=f"translation_{language.lower()}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                with st.expander(" Side by Side Comparison"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("**Original (English)**")
                        st.write(input_text)
                    with c2:
                        st.caption(f"**{language} Translation**")
                        st.write(translation)

                if use_context and context_used:
                    with st.expander(" Context used for translation"):
                        for i, chunk in enumerate(context_used[:2]):
                            st.caption(f"Reference {i+1} | Relevance: {chunk['similarity']}")
                            st.text(chunk["text"][:300])
                            st.divider()

            except Exception as e:
                st.error(f"Translation failed: {str(e)}")
                st.info("Make sure Ollama is running: `ollama serve`")

    # Simplify result
    if simplify_btn and input_text.strip():
        st.divider()
        with st.spinner("Simplifying text... (10-30 seconds)"):
            try:
                pipeline = get_pipeline(settings)
                translator = pipeline["translator"]

                result = translator.simplify(input_text)
                simplified = result["simplified"]

                st.subheader(" Plain English Version")
                st.text_area(
                    "Simplified output",
                    value=simplified,
                    height=200,
                    label_visibility="collapsed"
                )

                st.download_button(
                    label=" Download Simplified Version",
                    data=f"ORIGINAL:\n{input_text}\n\nSIMPLIFIED:\n{simplified}",
                    file_name="simplified.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                with st.expander(" Side by Side Comparison"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("**Original (Complex)**")
                        st.write(input_text)
                    with c2:
                        st.caption("**Simplified**")
                        st.write(simplified)

            except Exception as e:
                st.error(f"Simplification failed: {str(e)}")
                st.info("Make sure Ollama is running: `ollama serve`")

    # Quick test snippets
    st.divider()
    st.subheader(" Quick Test Snippets")
    st.write("Click any snippet to auto-fill the text box:")

    snippets = [
        "The Transformer model architecture relies entirely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
        "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        "The encoder maps an input sequence of symbol representations to a sequence of continuous representations."
    ]

    for snippet in snippets:
        if st.button(f"{snippet[:80]}...", use_container_width=True):
            st.session_state["prefill_text"] = snippet
            st.rerun()