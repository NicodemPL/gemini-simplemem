import os

# ============================================================================
# LLM Configuration
# ============================================================================

# API Key
OPENAI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

# LLM Model (Updated to gemini-2.5-flash based on availability)
LLM_MODEL = "gemini/gemini-2.5-flash"

# Embedding model
EMBEDDING_MODEL = "gemini/text-embedding-004"
EMBEDDING_DIMENSION = 768
EMBEDDING_CONTEXT_LENGTH = 8192

OPENAI_BASE_URL = None

# ============================================================================
# Advanced LLM Features
# ============================================================================

ENABLE_THINKING = False
USE_STREAMING = True
USE_JSON_FORMAT = True 

# ============================================================================
# Memory Building Parameters
# ============================================================================

WINDOW_SIZE = 40
OVERLAP_SIZE = 2

# ============================================================================
# Retrieval Parameters
# ============================================================================

SEMANTIC_TOP_K = 10
KEYWORD_TOP_K = 5
STRUCTURED_TOP_K = 5

# ============================================================================
# Database Configuration
# ============================================================================

# Use absolute path to ensure database is found regardless of where script is run
LANCEDB_PATH = "/Users/nicodem/Downloads/SimpleMem/lancedb_data"
MEMORY_TABLE_NAME = "memory_entries"

# ============================================================================
# Parallel Processing Configuration
# ============================================================================

ENABLE_PARALLEL_PROCESSING = False
MAX_PARALLEL_WORKERS = 4
ENABLE_PARALLEL_RETRIEVAL = False
MAX_RETRIEVAL_WORKERS = 4
ENABLE_PLANNING = True
ENABLE_REFLECTION = True
MAX_REFLECTION_ROUNDS = 2

# ============================================================================
# Judge Configuration
# ============================================================================

JUDGE_API_KEY = OPENAI_API_KEY
JUDGE_MODEL = LLM_MODEL
JUDGE_BASE_URL = None
JUDGE_ENABLE_THINKING = False
JUDGE_USE_STREAMING = False
JUDGE_TEMPERATURE = 0.3
