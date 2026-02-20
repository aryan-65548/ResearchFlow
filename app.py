import streamlit as st
from ui.sidebar import render_sidebar
from ui.upload import render_upload_page
from ui.chat import render_chat_page
from ui.translate import render_translate_page

st.set_page_config(
    page_title="Research Paper Translator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "processed_papers" not in st.session_state:
    st.session_state.processed_papers = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title(" Research Paper Translator")
st.caption("Powered by Ollama + ChromaDB + LangChain — 100% local, 100% private")

# Welcome banner — only show if no papers loaded
if not st.session_state.processed_papers:
    st.info("""
     Welcome! Here's how to get started:
    1. ⚙️ Configure your model and language in the **sidebar**
    2.  Upload a research paper PDF in the **Upload tab**
    3.  Ask questions in the **Chat tab**
    4.  Translate sections in the **Translate tab**
    """)

# Render sidebar
settings = render_sidebar()

# Tabs
tab1, tab2, tab3 = st.tabs([" Upload", " Chat", " Translate"])

with tab1:
    render_upload_page(settings)

with tab2:
    render_chat_page(settings)

with tab3:
    render_translate_page(settings)