# Chat Handler for RAG Chatbot
# Handles question answering logic

import re
import streamlit as st
import logging
from google.api_core.exceptions import ResourceExhausted
from .utils import remove_repetition, truncate_context, truncate_response, logger
from .reranker import rerank_documents, load_reranker


def format_answer_markdown(answer: str) -> str:
    """
    Format answer with proper markdown for better display.
    Convert inline lists to proper bullet points with newlines.
    """
    if not answer:
        return answer

    logger.info(f"[format_answer_markdown] Input: {answer[:100]}...")

    # Main pattern: Convert ". - " to newline bullet (handles most cases)
    # Matches: ". - ", ".- ", ". -", ".-"
    answer = re.sub(r"\.\s*-\s*", ".\n- ", answer)

    # Pattern for colon followed by dash: ": - " -> ":\n- "
    answer = re.sub(r":\s*-\s+", ":\n- ", answer)

    # Clean up multiple newlines
    answer = re.sub(r"\n{3,}", "\n\n", answer)

    # Ensure bullet points have proper spacing after dash
    answer = re.sub(r"\n-([^\s\n])", r"\n- \1", answer)

    # Fix double newlines before bullets
    answer = re.sub(r"\n\n-\s", "\n- ", answer)

    logger.info(f"[format_answer_markdown] Output: {answer[:100]}...")

    return answer.strip()


def _clean_answer(raw_output: str) -> str:
    """
    Clean model output to extract only the actual answer.
    Remove prompt echoes and extract content after answer markers.
    """
    logger.info(f"[_clean_answer] Raw output length: {len(raw_output)}")
    logger.debug(f"[_clean_answer] Raw output preview: {raw_output[:500]}...")

    answer = raw_output.strip()

    # List of markers that indicate the start of the actual answer
    # Order matters - check more specific patterns first
    answer_markers = [
        "### TRẢ LỜI (bằng tiếng Việt):",
        "### TRẢ LỜI:",
        "TRẢ LỜI:",
        "Trả lời:",
        "Câu trả lời:",
        "Answer:",
    ]

    # Find the FIRST occurrence of answer marker that appears AFTER the context
    # (to avoid matching markers within the echoed prompt)
    # Look for marker that has reasonable content after it (>100 chars)
    best_marker_pos = -1
    best_marker_len = 0

    for marker in answer_markers:
        pos = 0
        while True:
            pos = answer.find(marker, pos)
            if pos == -1:
                break

            # Check content length after this marker
            content_after = answer[pos + len(marker) :].strip()

            # If content after marker is reasonable (100-2000 chars), use this marker
            # Skip if content is too short (likely truncated) or too long (likely includes more prompt)
            if 50 < len(content_after) < 2500:
                if best_marker_pos == -1 or pos > best_marker_pos:
                    best_marker_pos = pos
                    best_marker_len = len(marker)
                break
            pos += 1

    # If no good marker found, try last occurrence as fallback
    if best_marker_pos == -1:
        for marker in answer_markers:
            pos = answer.rfind(marker)
            if pos > best_marker_pos:
                best_marker_pos = pos
                best_marker_len = len(marker)

    # Extract answer after the marker
    if best_marker_pos >= 0:
        answer = answer[best_marker_pos + best_marker_len :].strip()
        logger.info(
            f"[_clean_answer] Found marker at pos {best_marker_pos}, extracted {len(answer)} chars"
        )

    # Remove common prompt artifacts and headers
    prompt_artifacts = [
        "### NGỮ CẢNH:",
        "### CÂU HỎI:",
        "### HƯỚNG DẪN:",
        "Ngữ cảnh:",
        "Câu hỏi:",
        "Trả lời ngắn gọn dựa trên ngữ cảnh.",
        "Chỉ trả lời nội dung từ ngữ cảnh, không giải thích thêm.",
        'Nếu không có, nói "Không có trong tài liệu".',
        "Giải thích:",
        "Explanation:",
    ]

    for artifact in prompt_artifacts:
        if answer.startswith(artifact):
            answer = answer[len(artifact) :].strip()
            logger.info(f"[_clean_answer] Removed artifact: {artifact[:30]}...")

    # Stop at certain markers that indicate the model is rambling
    stop_markers = [
        "Giải thích:",
        "Explanation:",
        "Hy vọng",
        "Mời bạn",
        "Cảm ơn bạn",
        "Nếu bạn cần thêm",
        "Tôi hy vọng",
        "Tôi sẵn sàng",
        "(Trong trường hợp này",
        "(Maybe I can",
        "(Tôi có thể",
        "Hãy cho tôi biết",
        "Rất tiếc nếu",
    ]

    for marker in stop_markers:
        pos = answer.find(marker)
        if pos > 50:  # Only cut if we have some content before
            answer = answer[:pos].strip()
            logger.info(f"[_clean_answer] Cut at stop marker: {marker}")
            break

    # Clean leading punctuation
    answer = answer.lstrip(":;-–—•").strip()

    # Truncate if still too long (max ~500 chars for concise answer)
    if len(answer) > 800:
        # Find a good break point
        break_point = answer.rfind(". ", 0, 800)
        if break_point > 200:
            answer = answer[: break_point + 1]
            logger.info(f"[_clean_answer] Truncated to {len(answer)} chars")

    logger.info(f"[_clean_answer] Final answer length: {len(answer)}")

    return answer


def process_question(question: str, settings: dict, use_reranker: bool = True) -> tuple:
    """
    Process a question and generate answer using RAG.

    Args:
        question: User's question
        settings: Settings dict with model parameters
        use_reranker: Whether to use reranker for better retrieval

    Returns:
        tuple: (answer, sources)
    """
    logger.info(f"[process_question] Question: {question[:100]}...")
    logger.info(
        f"[process_question] Settings: num_chunks={settings['num_chunks']}, reranker={use_reranker}"
    )

    # Retrieve more documents initially for reranking
    initial_k = settings["num_chunks"] * 2 if use_reranker else settings["num_chunks"]
    st.session_state.retriever.search_kwargs["k"] = initial_k

    # Retrieve documents
    docs = st.session_state.retriever.invoke(question)
    logger.info(f"[process_question] Retrieved {len(docs)} documents (initial)")

    # Apply reranking if enabled
    if use_reranker and len(docs) > 1:
        docs, scores = rerank_documents(
            query=question,
            documents=docs,
            top_k=settings["num_chunks"],
            relevance_threshold=-5.0,  # Cross-encoder scores can be negative
        )
        logger.info(f"[process_question] After reranking: {len(docs)} documents")

    # Format context with source markers for traceability
    context_parts = []
    for i, doc in enumerate(docs):
        source_file = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"[Nguồn {i+1} - Trang {page}]: {doc.page_content}")

    context = "\n\n".join(context_parts)

    # Truncate context to avoid token overflow
    context = truncate_context(context, max_chars=10000)

    # Get sources info
    sources = []
    for doc in docs:
        page = doc.metadata.get("page", "N/A")
        start_index = doc.metadata.get("start_index", "N/A")
        sources.append(
            {
                "page": page + 1 if isinstance(page, int) else page,
                "start_index": start_index,
                "content": doc.page_content,
            }
        )

    # Generate answer
    mode = "Gemini" if st.session_state.is_gemini_mode else "Local (Qwen)"
    logger.info(f"[process_question] Generating answer using {mode} mode")

    try:
        if st.session_state.is_gemini_mode:
            answer = _generate_gemini_answer(context, question)
        else:
            answer = _generate_local_answer(context, question, settings)
    except ResourceExhausted as e:
        # Log the error
        logging.error(f"Google Generative AI API quota exceeded: {e}")
        
        # Notify the user (optional: you can customize this for your app's UI)
        logging.info("Falling back to local CPU-based processing.")

        # Fallback to CPU-based processing
        answer = _generate_local_answer(context, question, settings)

    # Post-process: Remove repetitive content
    answer = remove_repetition(answer)

    # Final truncation for very long answers
    answer = truncate_response(answer, max_sentences=8)

    # Format answer with proper markdown
    answer = format_answer_markdown(answer)

    logger.info(f"[process_question] Final answer: {answer[:200]}...")

    return answer, sources


def _generate_gemini_answer(context: str, question: str) -> str:
    """Generate answer using Gemini API"""
    logger.info("[_generate_gemini_answer] Calling Gemini API...")
    prompt_text = st.session_state.prompt.format(context=context, question=question)
    logger.debug(f"[_generate_gemini_answer] Prompt length: {len(prompt_text)}")

    response = st.session_state.llm.invoke(prompt_text)

    # Extract content from AIMessage
    raw_answer = response.content if hasattr(response, "content") else str(response)
    logger.info(f"[_generate_gemini_answer] Raw response length: {len(raw_answer)}")

    answer = _clean_answer(raw_answer)

    return answer


def _generate_local_answer(context: str, question: str, settings: dict) -> str:
    """Generate answer using local model (Qwen)"""
    logger.info("[_generate_local_answer] Calling local Qwen model...")

    # Update pipeline params
    st.session_state.llm.pipeline._forward_params.update(
        {
            "max_new_tokens": settings["max_new_tokens"],
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "repetition_penalty": settings["repetition_penalty"],
            "do_sample": True,
        }
    )
    logger.info(
        f"[_generate_local_answer] Params: max_tokens={settings['max_new_tokens']}, temp={settings['temperature']}"
    )

    # Format prompt
    prompt_text = st.session_state.prompt.format(context=context, question=question)
    logger.debug(f"[_generate_local_answer] Prompt length: {len(prompt_text)}")

    # Get answer directly from LLM
    output = st.session_state.llm.invoke(prompt_text)
    logger.info(f"[_generate_local_answer] Raw output length: {len(output)}")

    # Clean and extract answer
    answer = _clean_answer(output)

    return answer
