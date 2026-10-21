import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_DB_DIR = DATA_DIR / "vectordb"
METADATA_FILE = DATA_DIR / "metadata.json"

# Ensure essential directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# API Keys and Models
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "gemini")

# Chunking Configuration
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200

# Backend Server Details
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))
API_BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
