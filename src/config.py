"""
Configuration settings for the RAG application.
"""

# Paths
DATA_PATH = "data/"
CHROMA_PATH = "chroma/"
TRACKER_PATH = "document_tracker.json"

# Ollama configuration
OLLAMA_MODEL = "DeepSeek-R1:14b"
OLLAMA_URL = "http://localhost:11434"

# Text splitting configuration
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

# Chunking configuration  
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Vector database configuration
BATCH_SIZE = 30
MAX_WORKERS = 3 