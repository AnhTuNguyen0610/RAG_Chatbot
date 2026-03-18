# Configuration settings for RAG Chatbot

# Model settings
# GPU mode: Qwen2.5-3B-Instruct (local, requires GPU) - supports Vietnamese well
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# CPU/API mode: Google Gemini
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# Embedding model (used for both modes)
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

# Device settings
# "cuda" = GPU with Qwen2.5-3B local
# "cpu" = Gemini API (fast, no GPU needed)
DEFAULT_DEVICE = "cuda"
AVAILABLE_DEVICES = ["cuda", "cpu"]

# Default generation parameters
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.2

# Retrieval settings
DEFAULT_NUM_CHUNKS = 5
DEFAULT_MAX_CONTEXT_CHARS = 10000  # FIX: was 3000 in config but 10000 used in code — unified here
DEFAULT_LAMBDA_MULT = 0.7

# Chunking settings
CHUNK_MIN_SIZE = 500
CHUNK_BUFFER_SIZE = 1
CHUNK_BREAKPOINT_THRESHOLD = 95

# UI settings
PAGE_TITLE = "RAG PDF Chatbot with LLMs assistant"
PAGE_ICON = "📄"
LAYOUT = "wide"

# Chroma DB settings
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "pdf_docs"