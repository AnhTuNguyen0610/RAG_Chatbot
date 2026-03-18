# PDF Processing utilities for RAG Chatbot

import os
import gc
import shutil
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from .config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_MIN_SIZE,
    CHUNK_BUFFER_SIZE,
    CHUNK_BREAKPOINT_THRESHOLD,
    DEFAULT_LAMBDA_MULT,
)


def cleanup_chroma_db():
    """Clean up ChromaDB persistence directory to free disk space"""
    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
        except Exception as e:
            st.warning(f"Could not clean up ChromaDB directory: {e}")
    gc.collect()


def process_pdf(uploaded_files, embeddings, num_chunks=5, use_semantic_chunking=True):
    """
    Process uploaded PDF files and create vector database

    Args:
        uploaded_files: Single file or list of Streamlit uploaded file objects
        embeddings: Embedding model
        num_chunks: Number of chunks to retrieve (default k value)
        use_semantic_chunking: If True, use SemanticChunker (requires many embedding calls).
                               If False, use RecursiveCharacterTextSplitter (faster, no embedding needed for splitting)

    Returns:
        tuple: (retriever, prompt, total_chunks_count, vector_db, file_names)
    """
    # Clean up previous ChromaDB to prevent memory buildup
    cleanup_chroma_db()

    # Handle both single file and multiple files
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    all_documents = []
    file_names = []
    tmp_paths = []

    # Process each PDF file
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            tmp_paths.append(tmp_file_path)

        # Load PDF
        loader = PyPDFLoader(file_path=tmp_file_path)
        documents = loader.load()

        # Add source file name to metadata
        for doc in documents:
            doc.metadata["source_file"] = uploaded_file.name

        all_documents.extend(documents)
        file_names.append(uploaded_file.name)

    # Choose splitter based on mode
    if use_semantic_chunking:
        # SemanticChunker - better quality but requires many embedding API calls
        # Best for GPU mode with local embeddings
        splitter = SemanticChunker(
            embeddings=embeddings,
            buffer_size=CHUNK_BUFFER_SIZE,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=CHUNK_BREAKPOINT_THRESHOLD,
            min_chunk_size=CHUNK_MIN_SIZE,
            add_start_index=True,
        )
    else:
        # RecursiveCharacterTextSplitter - faster, no embedding needed for splitting
        # Best for API mode to avoid rate limits
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    # Split documents - use all_documents from multiple files
    docs = splitter.split_documents(all_documents)

    # Create vector database (in-memory to avoid lock issues)
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    # Create retriever
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": num_chunks,
            "lambda_mult": DEFAULT_LAMBDA_MULT,
        },
    )

    # Create Vietnamese prompt template - Optimized for RAG with anti-hallucination
    prompt = ChatPromptTemplate.from_template(
        """Bạn là trợ lý AI Việt Nam. BẮT BUỘC trả lời bằng TIẾNG VIỆT. KHÔNG dùng tiếng Trung, tiếng Anh hay ngôn ngữ khác.

### NGỮ CẢNH:
{context}

### CÂU HỎI: {question}

### HƯỚNG DẪN:
1. TRẢ LỜI BẰNG TIẾNG VIỆT - TUYỆT ĐỐI KHÔNG dùng tiếng Trung (中文)
2. CHỈ sử dụng thông tin từ ngữ cảnh trên
3. KHÔNG bịa đặt hay thêm thông tin ngoài ngữ cảnh
4. Nếu không tìm thấy câu trả lời, nói "Không tìm thấy thông tin này trong tài liệu"
5. Trả lời ngắn gọn, có cấu trúc rõ ràng
6. Nếu liệt kê nhiều điểm, dùng format:
   - Điểm 1
   - Điểm 2

### TRẢ LỜI (bằng tiếng Việt):"""
    )

    # Clean up all temporary files
    for tmp_path in tmp_paths:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Force garbage collection
    gc.collect()

    return retriever, prompt, len(docs), vector_db, file_names


def format_docs_with_sources(docs):
    """
    Format documents with source information

    Args:
        docs: List of retrieved documents

    Returns:
        tuple: (formatted_context, sources_list)
    """
    formatted = "\n\n".join(doc.page_content for doc in docs)
    sources = []

    for doc in docs:
        page = doc.metadata.get("page", "N/A")
        start_index = doc.metadata.get("start_index", "N/A")
        sources.append(
            {
                "page": page + 1 if isinstance(page, int) else page,
                "start_index": start_index,
                "content": doc.page_content,  # Full content, no truncation
            }
        )

    return formatted, sources
