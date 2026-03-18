# UI Components and Styling for RAG Chatbot

import os
import json
import streamlit as st

from .config import (
    DEFAULT_NUM_CHUNKS,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_DEVICE,
    AVAILABLE_DEVICES,
)

# API Key cache file path
API_KEY_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".api_key_cache.json")


def load_cached_api_key():
    """Load API key from cache file if exists"""
    try:
        if os.path.exists(API_KEY_CACHE_FILE):
            with open(API_KEY_CACHE_FILE, "r") as f:
                data = json.load(f)
                return data.get("gemini_api_key", "")
    except Exception:
        pass
    return ""


def save_api_key_to_cache(api_key: str):
    """Save API key to cache file"""
    try:
        with open(API_KEY_CACHE_FILE, "w") as f:
            json.dump({"gemini_api_key": api_key}, f)
        return True
    except Exception:
        return False


def clear_cached_api_key():
    """Clear cached API key"""
    try:
        if os.path.exists(API_KEY_CACHE_FILE):
            os.remove(API_KEY_CACHE_FILE)
        return True
    except Exception:
        return False


def apply_custom_css():
    """Apply custom CSS styling to the app"""
    st.markdown(
        """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main container styling */
    .main {
        padding: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling - more compact */
    .main-header {
        background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(139, 92, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        position: relative;
    }
    
    .main-header p {
        margin-top: 0.5rem;
        opacity: 0.9;
        font-size: 0.95rem;
        position: relative;
    }
    
    /* Chat container */
    .chat-container {
        background: #f8fafc;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        min-height: 200px;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #e2e8f0;
    }
    
    /* Chat message bubbles */
    .chat-message {
        display: flex;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-message.user {
        justify-content: flex-end;
    }
    
    .chat-message.assistant {
        justify-content: flex-start;
    }
    
    .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    
    .user .message-avatar {
        background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
        margin-left: 10px;
        order: 2;
    }
    
    .assistant .message-avatar {
        background: linear-gradient(135deg, #10b981, #059669);
        margin-right: 10px;
    }
    
    .message-content {
        max-width: 80%;
        padding: 1rem 1.25rem;
        border-radius: 16px;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .user .message-content {
        background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .assistant .message-content {
        background: white;
        color: #334155;
        border: 1px solid #e2e8f0;
        border-bottom-left-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    /* Card styling - modern glass effect */
    .info-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.25rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid rgba(139, 92, 246, 0.2);
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    .success-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        padding: 1.25rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid #10b981;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    
    /* Answer box styling - cleaner */
    .answer-box {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        position: relative;
    }
    
    .answer-box::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #10b981, #059669);
        border-radius: 16px 0 0 16px;
    }
    
    .answer-box h3 {
        color: #10b981;
        margin-bottom: 1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Source card styling */
    .source-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    
    .source-page {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    .source-position {
        background: #6c757d;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
    }
    
    .source-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
        line-height: 1.5;
        max-height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* File uploader styling - compact trong sidebar */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 0.75rem;
        border-radius: 12px;
        border: 2px dashed #8b5cf6;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #0ea5e9;
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    }
    
    [data-testid="stFileUploader"] section {
        padding: 0.5rem;
    }
    
    [data-testid="stFileUploader"] section > div {
        padding-top: 0 !important;
    }
    
    /* Button styling - modern with hover effects */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
        background: linear-gradient(135deg, #0284c7 0%, #7c3aed 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Text input styling - chat-like */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.875rem 1rem;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        background: white;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8;
    }
    
    /* Chat input container */
    .chat-input-container {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        font-weight: bold;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Feature list */
    .feature-list {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .feature-item:last-child {
        border-bottom: none;
    }
    
    .feature-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render the main header"""
    st.markdown(
        """
    <div class="main-header">
        <h1>🤖 AI PDF Assistant</h1>
        <p>Trò chuyện với AI để trao đổi về nội dung tài liệu PDF của bạn</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_instructions():
    """Render usage instructions"""
    st.markdown(
        """
    <div class="info-card">
        <h3>🚀 How to Use</h3>
        <div class="feature-item">
            <span class="feature-icon">1️⃣</span>
            <span><strong>Upload PDF</strong> - Select your PDF file and click "Process PDF"</span>
        </div>
        <div class="feature-item">
            <span class="feature-icon">2️⃣</span>
            <span><strong>Ask Questions</strong> - Type your question about the document</span>
        </div>
        <div class="feature-item">
            <span class="feature-icon">3️⃣</span>
            <span><strong>Get Answers</strong> - Receive AI-generated answers with source references</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("✨ Features & Models", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
            **📝 Chung (cả 2 mode)**
            - **Embedding:** BKAI Bi-Encoder
            - **Chunking:** Semantic
            - **Vector DB:** ChromaDB
            """
            )
        with col2:
            st.markdown(
                """
            **🖥️ GPU Mode**
            - **LLM:** Qwen2.5-3B
            - Chạy local, offline
            - Nhanh nếu có GPU mạnh
            """
            )
        with col3:
            st.markdown(
                """
            **☁️ CPU Mode**  
            - **LLM:** Gemini API
            - Cần internet + API key
            - Nhanh, không cần GPU
            """
            )


def render_sidebar():
    """Render sidebar with configuration options"""
    # Logo/Brand 
    st.sidebar.markdown(
        """
    <div style="text-align: center; padding: 1rem 0.5rem; margin-bottom: 0.5rem;">
        <div style="background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 50%, #ec4899 100%); 
                    width: 60px; height: 60px; border-radius: 16px; margin: 0 auto 0.75rem auto;
                    display: flex; align-items: center; justify-content: center;
                    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.35);">
            <span style="font-size: 1.8rem;">🤖</span>
        </div>
        <h3 style="margin: 0; color: #1e293b; font-weight: 700;">AI Chatbot</h3>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.8rem;">PDF Assistant</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ===== HƯỚNG DẪN SỬ DỤNG - ĐẦU TIÊN =====
    st.sidebar.markdown(
        """
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                padding: 0.75rem; border-radius: 10px; margin-bottom: 1rem;
                border-left: 3px solid #0ea5e9;">
        <p style="margin: 0 0 0.5rem 0; font-weight: 600; color: #0369a1; font-size: 0.85rem;">📋 Hướng dẫn sử dụng</p>
        <ol style="margin: 0; padding-left: 1.2rem; color: #475569; font-size: 0.8rem; line-height: 1.6;">
            <li>Upload file PDF bên dưới</li>
            <li>Nhấn <b>⚡ Xử lý PDF</b> để phân tích</li>
            <li>Đặt câu hỏi về nội dung PDF</li>
        </ol>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ===== UPLOAD SECTION - SAU HƯỚNG DẪN =====
    st.sidebar.markdown("### 📤 Upload tài liệu")

    upload_files = st.sidebar.file_uploader(
        "Chọn file PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="Kéo thả hoặc click để chọn nhiều file PDF",
        key="pdf_uploader",
    )

    # Hiển thị danh sách files đã chọn
    if upload_files:
        st.sidebar.markdown(f"**📁 {len(upload_files)} file đã chọn:**")
        for i, f in enumerate(upload_files, 1):
            file_size = len(f.getvalue()) / 1024  # KB
            st.sidebar.markdown(
                f"<small>{i}. {f.name} ({file_size:.1f} KB)</small>",
                unsafe_allow_html=True,
            )

    process_btn = st.sidebar.button(
        "⚡ Xử lý PDF",
        use_container_width=True,
        disabled=not upload_files,
        key="process_btn",
    )

    st.sidebar.markdown("---")

    # Model Status
    if st.session_state.get("models_loaded", False):
        st.sidebar.success("✅ Models đã sẵn sàng!")
    else:
        st.sidebar.info("⏳ Đang tải models...")

    st.sidebar.markdown("---")

    # ===== DEVICE SETTINGS =====
    with st.sidebar.expander("⚙️ Cài đặt thiết bị", expanded=False):
        device = st.selectbox(
            "Chọn thiết bị",
            options=AVAILABLE_DEVICES,
            index=(
                AVAILABLE_DEVICES.index(DEFAULT_DEVICE)
                if DEFAULT_DEVICE in AVAILABLE_DEVICES
                else 0
            ),
            help="CUDA = GPU local (Qwen) | CPU = Gemini API",
        )

    # API Key input for CPU/Gemini mode
    gemini_api_key = None
    if device == "cpu":
        with st.sidebar.expander("🔑 Gemini API Key", expanded=True):
            st.markdown(
                "<p style='font-size: 0.8rem; color: #64748b;'>CPU mode sử dụng Gemini API</p>",
                unsafe_allow_html=True,
            )

            # Load cached API key
            cached_key = load_cached_api_key()

            if cached_key:
                gemini_api_key = cached_key
                masked_key = (
                    cached_key[:8] + "..." + cached_key[-4:]
                    if len(cached_key) > 12
                    else "***"
                )
                st.text_input("API Key", value=masked_key, disabled=True)
                if st.button("🗑️ Xóa key", use_container_width=True):
                    clear_cached_api_key()
                    st.rerun()
            else:
                gemini_api_key = st.text_input(
                    "API Key",
                    type="password",
                    placeholder="Nhập API key...",
                )
                if gemini_api_key:
                    if st.button("💾 Lưu", use_container_width=True):
                        if save_api_key_to_cache(gemini_api_key):
                            st.success("✅ Đã lưu!")
                            st.rerun()
    else:
        st.sidebar.info("🖥️ GPU mode: Qwen2.5-3B")

    # Retrieval Settings
    with st.sidebar.expander("📚 Cài đặt truy xuất", expanded=False):
        num_chunks = st.slider(
            "Số chunks (k)",
            min_value=1,
            max_value=10,
            value=DEFAULT_NUM_CHUNKS,
            help="Số lượng chunks truy xuất từ vector database",
        )

        max_context_chars = st.slider(
            "Max context",
            min_value=1000,
            max_value=10000,
            value=DEFAULT_MAX_CONTEXT_CHARS,
            step=500,
            help="Độ dài tối đa của context đưa vào LLM",
        )

        use_reranker = st.checkbox(
            "🎯 Sử dụng Reranker",
            value=True,
            help="Reranker giúp sắp xếp lại chunks theo độ liên quan, giảm hallucination",
        )

    # Model Settings
    with st.sidebar.expander("🤖 Cài đặt model", expanded=False):
        max_new_tokens = st.slider(
            "Max tokens",
            min_value=128,
            max_value=2048,
            value=DEFAULT_MAX_NEW_TOKENS,
            step=64,
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
        )

        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=DEFAULT_TOP_P,
            step=0.05,
        )

        repetition_penalty = st.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=DEFAULT_REPETITION_PENALTY,
            step=0.1,
        )

    # Chat control section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 Điều khiển Chat")
    if st.sidebar.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        if "chat_history" in st.session_state:
            st.session_state.chat_history = []
            st.rerun()

    return {
        "device": device,
        "gemini_api_key": gemini_api_key,
        "num_chunks": num_chunks,
        "max_context_chars": max_context_chars,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "upload_files": upload_files,
        "process_btn": process_btn,
        "use_reranker": use_reranker,
    }


def render_gpu_status(gpu_info, show_clear_button=True):
    """Render GPU memory status and clear button in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎮 GPU Memory")

    if gpu_info["available"]:
        # Memory usage bar color based on usage
        usage = gpu_info["usage_percent"]
        if usage < 50:
            color = "#28a745"  # Green
            status = "🟢"
        elif usage < 80:
            color = "#ffc107"  # Yellow
            status = "🟡"
        else:
            color = "#dc3545"  # Red
            status = "🔴"

        st.sidebar.markdown(
            f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border: 1px solid #dee2e6;">
            <p style="margin: 0 0 0.5rem 0; font-weight: bold; color: #333;">
                🖥️ {gpu_info["gpu_name"]}
            </p>
            <div style="background: #e9ecef; border-radius: 5px; height: 20px; margin-bottom: 0.5rem;">
                <div style="background: {color}; width: {usage}%; height: 100%; border-radius: 5px; transition: width 0.3s;"></div>
            </div>
            <p style="margin: 0; font-size: 0.85rem; color: #666;">
                {status} <strong>{usage}%</strong> used ({gpu_info["reserved_gb"]:.1f} / {gpu_info["total_gb"]:.1f} GB)
            </p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.8rem; color: #888;">
                Free: {gpu_info["free_gb"]:.1f} GB | Allocated: {gpu_info["allocated_gb"]:.1f} GB
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if show_clear_button:
            if st.sidebar.button(
                "🗑️ Clear GPU Cache", use_container_width=True, help="Free up GPU memory"
            ):
                return "clear_cache"

            if st.sidebar.button(
                "🔄 Reload Models",
                use_container_width=True,
                help="Unload and reload all models (clears all GPU memory)",
            ):
                return "reload_models"
    else:
        st.sidebar.warning("⚠️ CUDA not available. Using CPU.")

    return None


def render_document_info(total_chunks):
    """Render document info in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 1rem; border-radius: 10px; text-align: center;">
        <h4 style="color: white; margin: 0;">📊 Document Info</h4>
        <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0 0 0;">
            {} Chunks
        </p>
    </div>
    """.format(
            total_chunks
        ),
        unsafe_allow_html=True,
    )


def render_answer(answer):
    """Render the answer in a styled box"""
    st.markdown(
        f"""
    <div class="answer-box">
        <h3>🤖 Trả lời từ AI</h3>
        <div style="color: #334155; line-height: 1.8; font-size: 0.95rem;">{answer}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_chat_message(role: str, content: str, icon: str = None):
    """Render a single chat message bubble"""
    if role == "user":
        avatar = "👤"
        st.markdown(
            f"""
        <div class="chat-message user">
            <div class="message-content">{content}</div>
            <div class="message-avatar">{avatar}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        avatar = "🤖"
        st.markdown(
            f"""
        <div class="chat-message assistant">
            <div class="message-avatar">{avatar}</div>
            <div class="message-content">{content}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_metrics(chunks_retrieved, chunks_used, context_length):
    """Render retrieval metrics"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value">{chunks_retrieved}</div>
            <div class="metric-label">Chunks Retrieved</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value">{chunks_used}</div>
            <div class="metric-label">Chunks Used</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value">{context_length:,}</div>
            <div class="metric-label">Context Chars</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_sources(sources, pdf_file=None):
    """Render source chunks with page information from PDF"""
    st.markdown("### 📑 Retrieved Chunks (Nguồn trích dẫn)")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Group chunks by page
    pages_used = set()
    for source in sources:
        if isinstance(source["page"], int):
            pages_used.add(source["page"])

    if pages_used:
        pages_str = ", ".join([str(p) for p in sorted(pages_used)])
        st.markdown(
            f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <p style="color: white; margin: 0; font-size: 1.1rem;">
                📖 <strong>Câu trả lời được trích từ các trang:</strong> {pages_str}
            </p>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                Tổng số chunks sử dụng: {len(sources)}
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    for i, source in enumerate(sources, 1):
        page_num = source["page"]

        # Create expander with chunk info
        with st.expander(f"📄 Chunk {i} | Trang {page_num} trong PDF", expanded=False):
            # Header with page info
            st.markdown(
                f"""
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1rem;">
                <span style="background: #667eea; color: white; padding: 0.4rem 1rem; border-radius: 20px; font-weight: bold;">
                    📄 Trang {page_num}
                </span>
                <span style="background: #28a745; color: white; padding: 0.4rem 1rem; border-radius: 20px;">
                    📍 Vị trí ký tự: {source['start_index']}
                </span>
                <span style="background: #17a2b8; color: white; padding: 0.4rem 1rem; border-radius: 20px;">
                    📏 Độ dài: {len(source['content'])} ký tự
                </span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # Full content display
            st.markdown("**📝 Nội dung chunk:**")
            st.markdown(
                f"""
            <div class="source-content">{source['content']}</div>
            """,
                unsafe_allow_html=True,
            )

            # Page navigation hint
            st.markdown(
                f"""
            <div style="background: #fff3cd; padding: 0.8rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #ffc107;">
                <p style="margin: 0; color: #856404;">
                    💡 <strong>Tip:</strong> Mở file PDF và đến <strong>trang {page_num}</strong> để xác minh thông tin này.
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_loading_models():
    """Render loading message for models"""
    st.markdown(
        """
    <div class="info-card">
        <h3>🔄 Loading Models...</h3>
        <p>Please wait while we load the AI models. This may take a few minutes on first run.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_success_message(message):
    """Render success message"""
    st.markdown(
        f"""
    <div class="success-card">
        <h3>✅ Success!</h3>
        <p>{message}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_results_simple(sources, metrics_data, gpu_info=None):
    """
    Render results with statistics and chunks only (no PDF viewer)
    Also displays updated GPU memory info
    """
    # Get unique pages from sources
    pages_used = set()
    for source in sources:
        if isinstance(source["page"], int):
            pages_used.add(source["page"])

    pages_str = ", ".join([str(p) for p in sorted(pages_used)]) if pages_used else "N/A"

    # Summary header with statistics
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.2rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <p style="color: white; margin: 0; font-size: 1.1rem;">
                    📖 <strong>Nguồn trích dẫn từ các trang:</strong> {pages_str}
                </p>
                <p style="color: rgba(255,255,255,0.8); margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                    Sử dụng {len(sources)} chunks | {metrics_data['context_length']:,} ký tự context
                </p>
            </div>
            <div style="display: flex; gap: 0.5rem;">
                <span style="background: rgba(255,255,255,0.2); color: white; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.85rem;">
                    📊 {metrics_data['chunks_retrieved']} chunks retrieved
                </span>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # GPU Memory info after generation (if available)
    if gpu_info and gpu_info.get("available"):
        usage = gpu_info["usage_percent"]
        if usage < 50:
            color = "#28a745"
            status = "🟢"
        elif usage < 80:
            color = "#ffc107"
            status = "🟡"
        else:
            color = "#dc3545"
            status = "🔴"

        st.markdown(
            f"""
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #dee2e6;">
            <p style="margin: 0; font-size: 0.9rem; color: #333;">
                🎮 <strong>GPU Memory sau generation:</strong> {status} {usage}% ({gpu_info["reserved_gb"]:.1f} / {gpu_info["total_gb"]:.1f} GB) | Free: {gpu_info["free_gb"]:.1f} GB
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Retrieved Chunks section
    st.markdown("### 📑 Retrieved Chunks (Nguồn trích dẫn)")

    for i, source in enumerate(sources, 1):
        page_num = source["page"]

        with st.expander(f"📄 Chunk {i} | Trang {page_num}", expanded=(i == 1)):
            # Badges
            st.markdown(
                f"""
            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 0.8rem;">
                <span style="background: #667eea; color: white; padding: 0.3rem 0.7rem; border-radius: 15px; font-size: 0.8rem; font-weight: bold;">
                    📄 Trang {page_num}
                </span>
                <span style="background: #28a745; color: white; padding: 0.3rem 0.7rem; border-radius: 15px; font-size: 0.8rem;">
                    📍 Pos: {source['start_index']}
                </span>
                <span style="background: #17a2b8; color: white; padding: 0.3rem 0.7rem; border-radius: 15px; font-size: 0.8rem;">
                    📏 {len(source['content'])} chars
                </span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Content
            st.markdown(
                f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; font-family: monospace; font-size: 0.82rem; line-height: 1.5; max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word;">{source['content']}</div>
            """,
                unsafe_allow_html=True,
            )
