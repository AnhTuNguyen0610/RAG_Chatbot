# Reranker module for improving retrieval quality
# Uses cross-encoder to rerank retrieved chunks

import logging
from typing import List, Tuple
import torch

logger = logging.getLogger("RAG_Chatbot")

# Global reranker instance
_reranker = None
_reranker_tokenizer = None

# Vietnamese Reranker options (in order of preference)
VIETNAMESE_RERANKER_MODELS = [
    "itdainb/PhoRanker",  # Vietnamese reranker - best for Vietnamese
    "nguyenvulebinh/vi-mrc-base",  # Vietnamese MRC - good for QA
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # Multilingual - supports Vietnamese
]


def load_reranker(device: str = "cuda"):
    """
    Load cross-encoder reranker model.
    Prioritizes Vietnamese-specific models.
    """
    global _reranker, _reranker_tokenizer

    if _reranker is not None:
        return _reranker, _reranker_tokenizer

    from sentence_transformers import CrossEncoder

    # Try Vietnamese models first, fallback to multilingual
    for model_name in VIETNAMESE_RERANKER_MODELS:
        try:
            logger.info(f"[Reranker] Trying {model_name}...")
            _reranker = CrossEncoder(model_name, max_length=512, device=device)
            logger.info(f"[Reranker] ✅ Loaded {model_name} successfully")
            return _reranker, None
        except Exception as e:
            logger.warning(f"[Reranker] Failed to load {model_name}: {e}")
            continue

    # Final fallback to original English model
    try:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"[Reranker] Fallback to {model_name}...")
        _reranker = CrossEncoder(model_name, max_length=512, device=device)
        logger.info(f"[Reranker] ✅ Loaded fallback model")
        return _reranker, None
    except Exception as e:
        logger.error(f"[Reranker] All models failed: {e}")
        return None, None


def rerank_documents(
    query: str, documents: List, top_k: int = 3, relevance_threshold: float = 0.1
) -> Tuple[List, List[float]]:
    """
    Rerank documents using cross-encoder and filter by relevance.

    Args:
        query: User's question
        documents: List of retrieved documents
        top_k: Number of top documents to return
        relevance_threshold: Minimum score to keep document

    Returns:
        Tuple of (reranked_documents, scores)
    """
    if not documents:
        return [], []

    global _reranker

    # If reranker not available, return original docs
    if _reranker is None:
        logger.info("[Reranker] Not available, using original order")
        return documents[:top_k], [1.0] * min(len(documents), top_k)

    try:
        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]

        # Get scores from cross-encoder
        scores = _reranker.predict(pairs)

        # Create list of (doc, score) tuples
        doc_scores = list(zip(documents, scores))

        # Sort by score descending
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter by relevance threshold and take top_k
        filtered = [
            (doc, score) for doc, score in doc_scores if score >= relevance_threshold
        ][:top_k]

        if not filtered:
            # If all filtered out, return top document anyway
            filtered = [doc_scores[0]] if doc_scores else []

        reranked_docs = [doc for doc, _ in filtered]
        final_scores = [score for _, score in filtered]

        logger.info(
            f"[Reranker] Reranked {len(documents)} -> {len(reranked_docs)} docs"
        )
        logger.info(f"[Reranker] Scores: {[f'{s:.3f}' for s in final_scores]}")

        return reranked_docs, final_scores

    except Exception as e:
        logger.warning(f"[Reranker] Error during reranking: {e}")
        return documents[:top_k], [1.0] * min(len(documents), top_k)


def compute_relevance_score(query: str, text: str) -> float:
    """
    Compute relevance score between query and text.

    Args:
        query: User's question
        text: Document text

    Returns:
        Relevance score (0-1)
    """
    global _reranker

    if _reranker is None:
        return 0.5  # Default neutral score

    try:
        score = _reranker.predict([(query, text)])[0]
        # Normalize to 0-1 range (cross-encoder scores can be negative)
        normalized = max(0, min(1, (score + 10) / 20))
        return normalized
    except:
        return 0.5


def is_relevant(query: str, text: str, threshold: float = 0.2) -> bool:
    """
    Check if text is relevant to query.

    Args:
        query: User's question
        text: Document text
        threshold: Minimum score to consider relevant

    Returns:
        True if relevant
    """
    global _reranker

    if _reranker is None:
        return True  # Assume relevant if no reranker

    try:
        score = _reranker.predict([(query, text)])[0]
        return score >= threshold
    except:
        return True
