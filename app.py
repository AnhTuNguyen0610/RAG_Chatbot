# RAG PDF Chatbot - Main Application
# A beautiful AI-powered document Q&A system with Vietnamese support

import torch
import streamlit as st

# Local imports
from src.config import PAGE_TITLE, PAGE_ICON, LAYOUT
from src.models import (
    load_embedding_model,
    load_local_llm_model,
    load_gemini_model,
    get_gpu_memory_info,
    clear_gpu_memory,
    unload_models,
)
from src.pdf_processor import process_pdf
from src.state_manager import init_session_state, reset_pdf_state, add_chat_message
from src.chat_handler import process_question
from src.reranker import load_reranker
from src.ui_components import (
    apply_custom_css,
    render_header,
    render_sidebar,
    render_document_info,
    render_chat_message,
    render_loading_models,
    render_gpu_status,
)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

init_session_state()
apply_custom_css()

# ============================================================================
# Main UI Layout
# ============================================================================

render_header()
settings = render_sidebar()

gpu_info = get_gpu_memory_info()
gpu_action = render_gpu_status(gpu_info)

# Handle GPU sidebar actions
if gpu_action == "clear_cache":
    clear_gpu_memory()
    st.rerun()
elif gpu_action == "reload_models":
    unload_models()
    st.session_state.models_loaded = False
    st.session_state.embeddings = None
    st.session_state.llm = None
    st.session_state.tokenizer = None
    st.session_state.is_gemini_mode = False
    reset_pdf_state()
    st.rerun()

# ============================================================================
# Model Loading
# ============================================================================

selected_device = settings["device"]
is_cpu_mode = selected_device == "cpu"

# Auto-reload when device changes
if (
    st.session_state.models_loaded
    and st.session_state.current_device != selected_device
):
    st.info(f"🔄 Đang chuyển từ {st.session_state.current_device} sang {selected_device}...")
    unload_models()
    st.session_state.models_loaded = False
    st.session_state.embeddings = None
    st.session_state.llm = None
    st.session_state.tokenizer = None
    st.session_state.is_gemini_mode = False
    reset_pdf_state()
    st.rerun()

# Check if API key is needed for CPU/Gemini mode
if is_cpu_mode and not settings.get("gemini_api_key"):
    st.warning("⚠️ Vui lòng nhập Gemini API key trong sidebar để tiếp tục!")
    st.info("💡 Lấy API key miễn phí tại: https://aistudio.google.com/app/apikey")
    st.stop()

# FIX: Detect API key change and force reload llm.
# load_gemini_model uses @st.cache_resource, so changing the key alone does NOT
# recreate the cached llm — the old (expired/invalid) client keeps being reused.
# We compare the current key against the one stored at load time and force a reload.
current_api_key = settings.get("gemini_api_key", "")
stored_api_key = st.session_state.get("gemini_api_key", "")
if (
    is_cpu_mode
    and st.session_state.models_loaded
    and current_api_key
    and current_api_key != stored_api_key
):
    st.info("🔄 API key thay đổi, đang kết nối lại Gemini...")
    st.session_state.llm = None
    st.session_state.models_loaded = False
    st.rerun()

# Load models if not loaded
if not st.session_state.models_loaded:
    render_loading_models()

    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner(f"Loading Vietnamese Bi-Encoder ({embed_device})..."):
        st.session_state.embeddings = load_embedding_model(device=embed_device)

    if is_cpu_mode:
        with st.spinner("🔗 Kết nối Gemini API..."):
            # FIX: api_key is passed as a parameter — not leaked into os.environ
            st.session_state.llm, st.session_state.tokenizer = load_gemini_model(
                api_key=settings["gemini_api_key"]
            )
            st.session_state.is_gemini_mode = True
            st.session_state.gemini_api_key = settings["gemini_api_key"]
        st.success(f"✅ Sẵn sàng! Embedding: BKAI ({embed_device}) | LLM: Gemini API")
    else:
        with st.spinner("Loading Qwen2.5-3B model..."):
            st.session_state.llm, st.session_state.tokenizer = load_local_llm_model()
            st.session_state.is_gemini_mode = False

        if st.session_state.llm:
            st.success("✅ Sẵn sàng! Embedding: BKAI (GPU) | LLM: Qwen2.5-3B (GPU)")
        else:
            st.error("❌ Không thể load model. Vui lòng dùng CPU mode.")
            st.stop()

    # Load reranker and store in session_state (not a bare global variable)
    with st.spinner("Loading Reranker model..."):
        try:
            reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
            # FIX: load_reranker now uses @st.cache_resource; the result is stored
            # in session_state under "_reranker_model" so reranker.py helpers can
            # access it without relying on a module-level global variable.
            reranker = load_reranker(device=reranker_device)
            st.session_state["_reranker_model"] = reranker
            st.session_state.reranker_loaded = reranker is not None
        except Exception as e:
            st.warning(f"⚠️ Reranker không khả dụng: {e}")
            st.session_state["_reranker_model"] = None
            st.session_state.reranker_loaded = False

    st.session_state.models_loaded = True
    st.session_state.current_device = selected_device
    st.rerun()

# ============================================================================
# PDF Processing (upload from sidebar)
# ============================================================================

upload_files = settings.get("upload_files")
process_btn = settings.get("process_btn")

if process_btn and upload_files:
    if st.session_state.rag_chain:
        reset_pdf_state()
        clear_gpu_memory()

    with st.spinner(f"Đang xử lý {len(upload_files)} file PDF..."):
        (
            st.session_state.retriever,
            st.session_state.prompt,
            total_chunks,
            vector_db,
            file_names,
        ) = process_pdf(
            upload_files,
            st.session_state.embeddings,
            settings["num_chunks"],
            use_semantic_chunking=True,
        )
        st.session_state.total_chunks = total_chunks
        st.session_state.rag_chain = True
        st.session_state.vector_store = vector_db
        st.session_state.pdf_names = file_names
        st.session_state.chat_history = []

    st.sidebar.success(f"✅ {len(file_names)} files, {total_chunks} chunks")
    st.rerun()

# Display document info
if st.session_state.rag_chain:
    render_document_info(st.session_state.total_chunks)

# ============================================================================
# Main Chat Interface
# ============================================================================

if not st.session_state.rag_chain:
    st.markdown(
        """
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                border-radius: 20px; margin: 2rem 0; border: 2px dashed #cbd5e1;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📄</div>
        <h3 style="color: #334155; margin-bottom: 0.5rem;">Chưa có tài liệu</h3>
        <p style="color: #64748b; max-width: 400px; margin: 0 auto;">
            Upload file PDF trong sidebar để bắt đầu trò chuyện với AI.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

else:
    pdf_names = st.session_state.get("pdf_names", [])
    if pdf_names:
        files_str = (
            ", ".join(pdf_names)
            if len(pdf_names) <= 3
            else f"{pdf_names[0]} và {len(pdf_names)-1} files khác"
        )
        st.markdown(
            f"""
        <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
                    padding: 0.75rem 1rem; border-radius: 12px; margin-bottom: 1rem;
                    border: 1px solid #10b981;">
            <p style="margin: 0; color: #065f46; font-weight: 600;">
                ✅ Đã tải: {files_str} ({st.session_state.total_chunks} chunks)
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Render chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            render_chat_message(msg["role"], msg["content"])

    question = st.chat_input("Nhập câu hỏi về tài liệu...")

    if question:
        render_chat_message("user", question)
        add_chat_message("user", question)

        with st.spinner("🤖 Đang xử lý..."):
            use_reranker = settings.get("use_reranker", True)

            # FIX: Pass llm, retriever, prompt, and is_gemini_mode explicitly
            # so that process_question is decoupled from st.session_state.
            answer, sources = process_question(
                question=question,
                settings=settings,
                retriever=st.session_state.retriever,
                llm=st.session_state.llm,
                prompt=st.session_state.prompt,
                is_gemini_mode=st.session_state.is_gemini_mode,
                use_reranker=use_reranker,
            )

        render_chat_message("assistant", answer)
        add_chat_message("assistant", answer)

        st.session_state.last_sources = sources

        if sources:
            with st.expander("📖 Nguồn trích dẫn", expanded=False):
                for i, src in enumerate(sources, 1):
                    page = src.get("page", "N/A")
                    content = src.get("content", "")[:300]
                    st.markdown(f"**Nguồn {i}** (Trang {page}):")
                    st.markdown(f"> {content}...")
                    st.markdown("---")

# ============================================================================
# Footer
# ============================================================================

st.markdown(
    """
<div style="text-align: center; color: #94a3b8; padding: 2rem 1rem; margin-top: 2rem;">
    <p style="font-size: 0.8rem; margin: 0;">AI Chatbot - PDF RAG Assistant </p>
    <p style="font-size: 0.75rem; margin: 0.25rem 0 0 0;">Powered by BKAI Embedding & Qwen/Gemini LLM</p>
</div>
""",
    unsafe_allow_html=True,
)