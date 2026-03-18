# Utility functions for RAG Chatbot

import re
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAG_Chatbot")


def remove_repetition(text: str, threshold: float = 0.6) -> str:
    """
    Remove repetitive sentences/paragraphs from generated text.

    This helps fix the issue where models like Qwen can generate
    heavily repetitive output.

    Args:
        text: The generated text to clean
        threshold: Similarity threshold (0-1) above which sentences are considered duplicates

    Returns:
        Cleaned text with repetitions removed
    """
    if not text or len(text) < 50:
        return text

    logger.info(f"[remove_repetition] Input length: {len(text)}")

    # Split into sentences (handle Vietnamese punctuation)
    sentences = re.split(r"(?<=[.!?。])\s+", text)

    if len(sentences) <= 1:
        return text

    seen_sentences = []
    result_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Skip very short sentences (likely incomplete)
        if len(sentence) < 15:
            result_sentences.append(sentence)
            continue

        # Check if this sentence is similar to any seen sentence
        is_duplicate = False
        sentence_words = set(sentence.lower().split())

        for seen in seen_sentences:
            seen_words = set(seen.lower().split())

            # Calculate Jaccard similarity
            if len(sentence_words) > 0 and len(seen_words) > 0:
                intersection = len(sentence_words & seen_words)
                union = len(sentence_words | seen_words)
                similarity = intersection / union if union > 0 else 0

                if similarity > threshold:
                    is_duplicate = True
                    logger.debug(
                        f"[remove_repetition] Duplicate found (sim={similarity:.2f}): {sentence[:50]}..."
                    )
                    break

        if not is_duplicate:
            seen_sentences.append(sentence)
            result_sentences.append(sentence)

    # Join sentences back together
    cleaned_text = " ".join(result_sentences)

    # Also check for repeated phrases within the text
    # Pattern: same phrase repeated 2+ times consecutively
    cleaned_text = re.sub(r"(.{15,}?)\1{1,}", r"\1", cleaned_text)

    logger.info(f"[remove_repetition] Output length: {len(cleaned_text)}")

    return cleaned_text.strip()


def truncate_response(text: str, max_sentences: int = 8) -> str:
    """
    Truncate response to maximum number of sentences.
    Prevents overly long, rambling responses.

    Args:
        text: Response text
        max_sentences: Maximum sentences to keep

    Returns:
        Truncated text
    """
    if not text:
        return text

    # Split into sentences
    sentences = re.split(r"(?<=[.!?。])\s+", text)

    if len(sentences) <= max_sentences:
        return text

    # Keep only first max_sentences
    truncated = " ".join(sentences[:max_sentences])
    logger.info(
        f"[truncate_response] Truncated from {len(sentences)} to {max_sentences} sentences"
    )

    return truncated


def truncate_context(context: str, max_chars: int = 12000) -> str:
    """Truncate context to avoid token overflow"""
    if len(context) > max_chars:
        logger.info(
            f"[truncate_context] Truncated from {len(context)} to {max_chars} chars"
        )
        return context[:max_chars] + "\n... (đã cắt bớt)"
    return context
