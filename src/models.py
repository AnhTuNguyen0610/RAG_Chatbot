# Model loading utilities for RAG Chatbot

import gc
import torch
import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig
from .config import LOCAL_MODEL_NAME, GEMINI_MODEL_NAME, EMBEDDING_MODEL, DEFAULT_DEVICE

# ============================================================================
# GPU Memory Management
# ============================================================================

def get_gpu_memory_info():
    """Get GPU memory usage information"""
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        free_memory = total_memory - reserved_memory
        return {
            "available": True,
            "gpu_name": gpu_name,
            "total_gb": round(total_memory, 2),
            "allocated_gb": round(allocated_memory, 2),
            "reserved_gb": round(reserved_memory, 2),
            "free_gb": round(free_memory, 2),
            "usage_percent": round((reserved_memory / total_memory) * 100, 1),
        }
    return {"available": False}


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"Warning: Could not clear CUDA cache: {e}")
    return True


def clear_vector_store():
    """Clear ChromaDB vector store from memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Model Loading Functions
# ============================================================================

@st.cache_resource
def load_embedding_model(device: str = None):
    """Load and cache the embedding model"""
    device = device or DEFAULT_DEVICE

    if device == "auto":
        model_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        model_device = "cpu"
        st.warning("CUDA not available, falling back to CPU for embeddings")
    else:
        model_device = device

    model_kwargs = {"device": model_device}
    encode_kwargs = {"normalize_embeddings": True}

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def load_gemini_model(api_key: str):
    """
    Load Gemini model with API key.
    NOTE: Intentionally NOT cached with @st.cache_resource.
    The Gemini client is cheap to instantiate (no weights to load) and caching
    it would cause stale/expired keys to keep being used after the user updates
    their key — exactly the bug this fixes.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=api_key,
        temperature=0.7,
        max_output_tokens=1024,
    )
    return llm, None  # No tokenizer needed for API


def load_gemini_embeddings(api_key: str):
    """Load Google Gemini embedding model via API — fast, no GPU needed"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    return embeddings


@st.cache_resource
def load_local_llm_model():
    """Load and cache Qwen2.5-3B-Instruct model with 4-bit quantization for GPU"""
    if not torch.cuda.is_available():
        st.error("❌ GPU không khả dụng! Vui lòng chọn CPU mode để dùng Gemini API.")
        return None, None

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_NAME,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto",
        do_sample=True,
    )

    return HuggingFacePipeline(pipeline=model_pipeline), tokenizer


def unload_models():
    """Unload all cached models and clear memory"""
    st.cache_resource.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return True