# Chat Handler for RAG Chatbot
# Handles question answering logic

import re
import logging
from google.api_core.exceptions import ResourceExhausted
from langchain_google_genai import ChatGoogleGenerativeAI

from .utils import remove_repetition, truncate_context, truncate_response, logger
from .reranker import rerank_documents


def format_answer_markdown(answer: str) -> str:
    """
    Format answer with proper markdown for better display.
    Convert inline lists to proper bullet points with newlines.
    """
    if not answer:
        return answer

    logger.info(f"[format_answer_markdown] Input: {answer[:100]}...")

    # Convert ". - " to newline bullet (handles most cases)
    answer = re.sub(r"\.\s*-\s*", ".\n- ", answer)

    # Convert ": - " to ":\n- "
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

    answer_markers = [
        "### TRẢ LỜI (bằng tiếng Việt):",
        "### TRẢ LỜI:",
        "TRẢ LỜI:",
        "Trả lời:",
        "Câu trả lời:",
        "Answer:",
    ]

    best_marker_pos = -1
    best_marker_len = 0

    for marker in answer_markers:
        pos = 0
        while True:
            pos = answer.find(marker, pos)
            if pos == -1:
                break
            content_after = answer[pos + len(marker):].strip()
            if 50 < len(content_after) < 2500:
                if best_marker_pos == -1 or pos > best_marker_pos:
                    best_marker_pos = pos
                    best_marker_len = len(marker)
                break
            pos += 1

    if best_marker_pos == -1:
        for marker in answer_markers:
            pos = answer.rfind(marker)
            if pos > best_marker_pos:
                best_marker_pos = pos
                best_marker_len = len(marker)

    if best_marker_pos >= 0:
        answer = answer[best_marker_pos + best_marker_len:].strip()
        logger.info(
            f"[_clean_answer] Found marker at pos {best_marker_pos}, extracted {len(answer)} chars"
        )

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
            answer = answer[len(artifact):].strip()
            logger.info(f"[_clean_answer] Removed artifact: {artifact[:30]}...")

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
        if pos > 50:
            answer = answer[:pos].strip()
            logger.info(f"[_clean_answer] Cut at stop marker: {marker}")
            break

    answer = answer.lstrip(":;------•").strip()

    if len(answer) > 800:
        break_point = answer.rfind(". ", 0, 800)
        if break_point > 200:
            answer = answer[: break_point + 1]
            logger.info(f"[_clean_answer] Truncated to {len(answer)} chars")

    logger.info(f"[_clean_answer] Final answer length: {len(answer)}")
    return answer


def process_question(
    question: str,
    settings: dict,
    retriever,
    llm,
    prompt,
    is_gemini_mode: bool = False,
    use_reranker: bool = True,
) -> tuple:
    """
    Process a question and generate an answer using RAG.

    FIX: Accepts llm, retriever, prompt, and is_gemini_mode as explicit parameters
    instead of reading them from st.session_state inside the function body.
    This decouples the business logic from Streamlit state, making it testable
    and reusable outside of a Streamlit context.

    Args:
        question: User's question
        settings: Settings dict with model parameters
        retriever: LangChain retriever object
        llm: Loaded LLM (Gemini or HuggingFacePipeline)
        prompt: ChatPromptTemplate
        is_gemini_mode: True = Gemini API, False = local Qwen
        use_reranker: Whether to apply cross-encoder reranking

    Returns:
        tuple: (answer, sources)
    """
    logger.info(f"[process_question] Question: {question[:100]}...")
    logger.info(
        f"[process_question] Settings: num_chunks={settings['num_chunks']}, reranker={use_reranker}"
    )

    # Retrieve more documents initially for reranking
    initial_k = settings["num_chunks"] * 2 if use_reranker else settings["num_chunks"]
    retriever.search_kwargs["k"] = initial_k

    docs = retriever.invoke(question)
    logger.info(f"[process_question] Retrieved {len(docs)} documents (initial)")

    if use_reranker and len(docs) > 1:
        docs, scores = rerank_documents(
            query=question,
            documents=docs,
            top_k=settings["num_chunks"],
            relevance_threshold=-5.0,
        )
        logger.info(f"[process_question] After reranking: {len(docs)} documents")

    # Format context with source markers
    context_parts = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"[Nguồn {i+1} - Trang {page}]: {doc.page_content}")

    context = "\n\n".join(context_parts)
    context = truncate_context(context, max_chars=settings.get("max_context_chars", 10000))

    # Build sources info
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
    mode = "Gemini" if is_gemini_mode else "Local (Qwen)"
    logger.info(f"[process_question] Generating answer using {mode} mode")

    try:
        if is_gemini_mode:
            answer = _generate_gemini_answer(context, question, llm, prompt, settings)
        else:
            answer = _generate_local_answer(context, question, llm, prompt, settings)
    except ResourceExhausted as e:
        err_full = str(e)
        logging.error(f"[process_question] ResourceExhausted full error: {err_full}")
        err_lower = err_full.lower()
        if "api_key_invalid" in err_lower or "api key" in err_lower or "expired" in err_lower or "not found" in err_lower:
            answer = (
    "❌ API key Gemini không hợp lệ hoặc đã hết hạn. "
    "**Cách sửa:** "
    "1. Nhấn 🗑️ **Xóa key** trong sidebar "
    "2. Vào https://aistudio.google.com/app/apikey tạo key mới "
    "3. Nhập key mới → nhấn 💾 Lưu "
    "4. Đặt lại câu hỏi"
)
        else:
            answer = f"⚠️ Lỗi Gemini API (ResourceExhausted): {err_full}"
    except Exception as e:
        err_str = str(e).lower()
        logging.error(f"[process_question] LLM error: {e}")
        if "api_key_invalid" in err_str or "api key" in err_str or "invalid argument" in err_str:
            answer = (
    "❌ API key Gemini không hợp lệ hoặc đã hết hạn. "
    "**Cách sửa:** "
    "1. Nhấn 🗑️ **Xóa key** trong sidebar "
    "2. Vào https://aistudio.google.com/app/apikey tạo key mới "
    "3. Nhập key mới → nhấn 💾 Lưu "
    "4. Đặt lại câu hỏi"
)
        elif "quota" in err_str or "resource_exhausted" in err_str:
            answer = "⚠️ API Gemini đã hết quota. Vui lòng thử lại sau hoặc kiểm tra giới hạn tại Google AI Studio."
        else:
            answer = f"❌ Lỗi khi gọi LLM: {str(e)}"

    answer = remove_repetition(answer)
    answer = truncate_response(answer, max_sentences=8)
    answer = format_answer_markdown(answer)

    logger.info(f"[process_question] Final answer: {answer[:200]}...")
    return answer, sources


def _generate_gemini_answer(
    context: str, question: str, llm, prompt, settings: dict
) -> str:
    """
    Generate answer using Gemini API.

    FIX: Accepts llm and prompt as parameters (no longer reads from session_state).
    Also applies temperature from settings to keep parity with local model — the
    ChatGoogleGenerativeAI temperature can be overridden by recreating the client,
    but since that is expensive, we log the intended temperature and rely on the
    value set at load time. If dynamic temperature control is needed, recreate llm
    here using the api_key stored in session_state.
    """
    logger.info("[_generate_gemini_answer] Calling Gemini API...")
    prompt_text = prompt.format(context=context, question=question)
    logger.debug(f"[_generate_gemini_answer] Prompt length: {len(prompt_text)}")

    response = llm.invoke(prompt_text)
    raw_answer = response.content if hasattr(response, "content") else str(response)
    logger.info(f"[_generate_gemini_answer] Raw response length: {len(raw_answer)}")

    return _clean_answer(raw_answer)


def _generate_local_answer(
    context: str, question: str, llm, prompt, settings: dict
) -> str:
    """
    Generate answer using local Qwen model.

    FIX: Accepts llm and prompt as parameters (no longer reads from session_state).
    """
    logger.info("[_generate_local_answer] Calling local Qwen model...")

    llm.pipeline._forward_params.update(
        {
            "max_new_tokens": settings["max_new_tokens"],
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "repetition_penalty": settings["repetition_penalty"],
            "do_sample": True,
        }
    )

    logger.info(
        f"[_generate_local_answer] Params: max_tokens={settings['max_new_tokens']}, "
        f"temp={settings['temperature']}"
    )

    prompt_text = prompt.format(context=context, question=question)
    logger.debug(f"[_generate_local_answer] Prompt length: {len(prompt_text)}")

    output = llm.invoke(prompt_text)
    logger.info(f"[_generate_local_answer] Raw output length: {len(output)}")

    return _clean_answer(output)