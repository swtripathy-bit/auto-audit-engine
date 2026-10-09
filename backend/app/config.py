"""
Configuration and Environment Settings.
Loads configuration and API keys from environment or .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Search for .env in project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


class Settings:
    PROJECT_NAME: str = "Autonomous Data Auditing & ML-Readiness Engine"
    API_V1_STR: str = "/api/v1"
    
    # Gemini API settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    
    # Backend server settings
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))


settings = Settings()
