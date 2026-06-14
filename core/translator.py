import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from core.retriever import Retriever
from dotenv import load_dotenv

load_dotenv()

class Translator:
    """
    Connects Groq LLM with the RAG retriever to:
    1. Answer questions about a research paper
    2. Translate sections of the paper into any language
    
    Uses retrieved context chunks so answers are grounded
    in the actual paper content — not LLM hallucination.
    """

    # Languages supported for translation
    SUPPORTED_LANGUAGES = [
        "Hindi", "Gujarati", "Spanish", "French", "German",
        "Chinese", "Japanese", "Arabic", "Portuguese", "Italian",
        "Korean", "Russian", "Dutch", "Turkish", "Bengali"
    ]

    def __init__(
        self,
        retriever: Retriever,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: str = None,
        temperature: float = 0.3
    ):
        """
        retriever: your Retriever instance (from Day 5)
        model_name: which Groq-hosted model to use
                     (e.g. "llama-3.3-70b-versatile", "llama-3.1-8b-instant")
        api_key: Groq API key. If not provided, read from GROQ_API_KEY env var.
        temperature: 0.0 = focused/deterministic, 1.0 = creative
                     0.3 is good for translation (accurate but natural)
        """
        self.retriever = retriever

        api_key = api_key or os.getenv("GROQ_API_KEY")

        # Fallback for Streamlit Community Cloud secrets (st.secrets)
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your .env file, environment "
                "variables, or Streamlit secrets."
            )

        print(f"Connecting to Groq model: {model_name}")
        self.llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature
        )
        print("Groq connected!")

    def answer_question(self, question: str) -> dict:
        """
        Takes a question, retrieves relevant chunks,
        sends them to Groq as context, returns the answer.

        Returns dict with:
        - answer: the LLM's response
        - context_used: the chunks retrieved
        - avg_relevance: how relevant the context was
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        # Step 1: Retrieve relevant context from ChromaDB
        chunks, avg_relevance = self.retriever.retrieve_with_scores(question)

        # Step 2: Check if paper is even relevant to this question
        if avg_relevance < 0.25:
            return {
                "answer": "I couldn't find relevant information in the paper to answer this question.",
                "context_used": [],
                "avg_relevance": avg_relevance
            }

        # Step 3: Format context for the prompt
        context = self.retriever.retrieve_as_context(question)

        # Step 4: Build the prompt
        # System message tells LLM its role and rules
        system_message = SystemMessage(content="""You are an expert research paper assistant.
Your job is to answer questions about research papers clearly and accurately.

Rules:
- Only use the provided context to answer
- If the context doesn't contain the answer, say so honestly
- Use simple language to explain complex concepts
- Be concise but complete
- Always cite which context section your answer comes from""")

        # Human message contains the context + question
        human_message = HumanMessage(content=f"""Here is the relevant context from the research paper:

{context}

Based on the context above, please answer this question:
{question}""")

        # Step 5: Send to Groq and get response
        print(f"Sending to Groq... (this may take 10-30 seconds)")
        response = self.llm.invoke([system_message, human_message])

        return {
            "answer": response.content,
            "context_used": chunks,
            "avg_relevance": avg_relevance
        }

    def translate(
        self,
        text: str,
        target_language: str,
        use_context: bool = True
    ) -> dict:
        """
        Translates any text into the target language.
        If use_context=True, retrieves related paper sections
        to ensure translation uses correct technical terminology.

        Returns dict with:
        - translation: the translated text
        - target_language: language translated to
        - context_used: chunks used for terminology reference
        """
        if not text.strip():
            raise ValueError("Text to translate cannot be empty")

        if target_language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language not supported. Choose from: {', '.join(self.SUPPORTED_LANGUAGES)}"
            )

        # Get context for technical terminology consistency
        context = ""
        chunks = []
        if use_context:
            chunks, _ = self.retriever.retrieve_with_scores(text[:200])
            if chunks:
                # Only use top 2 chunks for translation context
                # (we don't need as much context as for QA)
                context_parts = [
                    f"[Reference {i+1}]: {c['text']}"
                    for i, c in enumerate(chunks[:2])
                ]
                context = "\n\n".join(context_parts)

        # Build translation prompt
        system_message = SystemMessage(content=f"""You are an expert scientific translator 
specializing in translating research papers into {target_language}.

Rules:
- Translate accurately and naturally into {target_language}
- Preserve all technical terms correctly
- Keep the academic tone of the original
- Do not add explanations — only translate
- If a technical term has no translation, keep it in English""")

        # Build human message based on whether we have context
        if context:
            human_content = f"""Use this reference context for correct technical terminology:

{context}

Now translate the following text into {target_language}:

{text}

Provide ONLY the translation, nothing else."""
        else:
            human_content = f"""Translate the following research paper text into {target_language}:

{text}

Provide ONLY the translation, nothing else."""

        human_message = HumanMessage(content=human_content)

        print(f"Translating to {target_language}... (10-30 seconds)")
        response = self.llm.invoke([system_message, human_message])

        return {
            "translation": response.content,
            "target_language": target_language,
            "context_used": chunks
        }

    def simplify(self, text: str) -> dict:
        """
        BONUS: Simplifies complex academic text into plain English.
        Great for making research papers accessible.
        """
        system_message = SystemMessage(content="""You are an expert at explaining 
complex research concepts in simple, everyday language.

Rules:
- Explain as if talking to a smart 16 year old
- Replace jargon with simple words
- Use analogies where helpful
- Keep it concise""")

        human_message = HumanMessage(content=f"""Simplify this research paper text 
into plain English:

{text}""")

        print("Simplifying text... (10-30 seconds)")
        response = self.llm.invoke([system_message, human_message])

        return {
            "simplified": response.content,
            "original": text
        }
