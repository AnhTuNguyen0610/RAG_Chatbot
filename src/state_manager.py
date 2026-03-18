# Session State Management for RAG Chatbot

import streamlit as st
from .models import clear_vector_store


def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "rag_chain": None,
        "models_loaded": False,
        "reranker_loaded": False,
        "embeddings": None,
        "llm": None,
        "retriever": None,
        "prompt": None,
        "tokenizer": None,
        "total_chunks": 0,
        "chat_history": [],
        "last_sources": [],  # Sources from last answer
        "pdf_bytes": None,
        "pdf_name": None,
        "pdf_names": [],  # List of uploaded file names
        "current_device": None,
        "vector_store": None,
        "is_gemini_mode": False,
        "gemini_api_key": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_pdf_state():
    """Reset PDF-related state (called when processing new PDF)"""
    # Clear vector store from memory
    if st.session_state.vector_store is not None:
        try:
            del st.session_state.vector_store
        except:
            pass
        st.session_state.vector_store = None

    # Clear retriever
    if st.session_state.retriever is not None:
        try:
            del st.session_state.retriever
        except:
            pass
        st.session_state.retriever = None

    # Clear GPU cache after removing old data
    clear_vector_store()

    # Reset related state
    st.session_state.rag_chain = None
    st.session_state.prompt = None
    st.session_state.total_chunks = 0
    st.session_state.pdf_bytes = None
    st.session_state.pdf_name = None
    st.session_state.pdf_names = []


def clear_chat_history():
    """Clear chat history"""
    st.session_state.chat_history = []


def add_chat_message(role: str, content: str):
    """Add a message to chat history"""
    st.session_state.chat_history.append({"role": role, "content": content})
