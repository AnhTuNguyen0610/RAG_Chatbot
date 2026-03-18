# Reranker module for improving retrieval quality
# Uses cross-encoder to rerank retrieved chunks

import logging
from typing import List, Tuple

import torch
import streamlit as st

logger = logging.getLogger("RAG_Chatbot")

# Vietnamese Reranker options (in order of preference)
VIETNAMESE_RERANKER_MODELS = [
    "itdainb/PhoRanker",                             # Vietnamese reranker — best for Vietnamese
    "nguyenvulebinh/vi-mrc-base",                    # Vietnamese MRC — good for QA
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",   # Multilingual — supports Vietnamese
]


@st.cache_resource
def load_reranker(device: str = "cuda"):
    """
    Load cross-encoder reranker model and cache it with Streamlit.
    FIX: Was using a bare module-level global variable which is unsafe across
    Streamlit reruns and multi-threading. Using @st.cache_resource ensures the
    model is loaded once and shared safely across all sessions.

    Prioritizes Vietnamese-specific models, falls back to multilingual / English.

    Returns:
        CrossEncoder instance, or None if all models fail to load.
    """
    from sentence_transformers import CrossEncoder

    for model_name in VIETNAMESE_RERANKER_MODELS:
        try:
            logger.info(f"[Reranker] Trying {model_name}...")
            reranker = CrossEncoder(model_name, max_length=512, device=device)
            logger.info(f"[Reranker] ✅ Loaded {model_name} successfully")
            return reranker
        except Exception as e:
            logger.warning(f"[Reranker] Failed to load {model_name}: {e}")
            continue

    # Final fallback to English model
    try:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"[Reranker] Fallback to {model_name}...")
        reranker = CrossEncoder(model_name, max_length=512, device=device)
        logger.info("[Reranker] ✅ Loaded fallback model")
        return reranker
    except Exception as e:
        logger.error(f"[Reranker] All models failed: {e}")
        return None


def rerank_documents(
    query: str,
    documents: List,
    top_k: int = 3,
    relevance_threshold: float = 0.1,
) -> Tuple[List, List[float]]:
    """
    Rerank documents using cross-encoder and filter by relevance.

    Args:
        query: User's question
        documents: List of retrieved documents
        top_k: Number of top documents to return
        relevance_threshold: Minimum score to keep a document

    Returns:
        Tuple of (reranked_documents, scores)
    """
    if not documents:
        return [], []

    # Retrieve the cached reranker (no global variable needed)
    reranker = st.session_state.get("_reranker_model")

    if reranker is None:
        logger.info("[Reranker] Not available, using original order")
        return documents[:top_k], [1.0] * min(len(documents), top_k)

    try:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)

        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        filtered = [
            (doc, score) for doc, score in doc_scores if score >= relevance_threshold
        ][:top_k]

        if not filtered:
            filtered = [doc_scores[0]] if doc_scores else []

        reranked_docs = [doc for doc, _ in filtered]
        final_scores = [score for _, score in filtered]

        logger.info(f"[Reranker] Reranked {len(documents)} -> {len(reranked_docs)} docs")
        logger.info(f"[Reranker] Scores: {[f'{s:.3f}' for s in final_scores]}")

        return reranked_docs, final_scores

    except Exception as e:
        logger.warning(f"[Reranker] Error during reranking: {e}")
        return documents[:top_k], [1.0] * min(len(documents), top_k)


def compute_relevance_score(query: str, text: str) -> float:
    """
    Compute a normalised relevance score (0–1) between query and text.
    """
    reranker = st.session_state.get("_reranker_model")
    if reranker is None:
        return 0.5

    try:
        score = reranker.predict([(query, text)])[0]
        return max(0.0, min(1.0, (score + 10) / 20))
    except Exception:
        return 0.5


def is_relevant(query: str, text: str, threshold: float = 0.2) -> bool:
    """
    Return True if text is considered relevant to the query.
    """
    reranker = st.session_state.get("_reranker_model")
    if reranker is None:
        return True  # Assume relevant if no reranker available

    try:
        score = reranker.predict([(query, text)])[0]
        return score >= threshold
    except Exception:
        return True