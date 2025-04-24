"""
Configuration module for Life-Block Backend.

This module handles all configuration settings for the application,
centralizing environment variables and default values.
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables with defaults.
    """
    # API and model configuration
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    model_id: str = os.environ.get("MODEL_ID", "gpt-4o")
    
    # Agent configuration
    max_steps: int = int(os.environ.get("MAX_STEPS", "7"))
    verbosity_level: int = int(os.environ.get("VERBOSITY_LEVEL", "1"))
    planning_interval: int = int(os.environ.get("PLANNING_INTERVAL", "3"))
    
    # Authorized imports for the code agent
    authorized_imports: List[str] = ["time", "numpy", "pandas"]
    
    # Application configuration
    debug: bool = os.environ.get("DEBUG", "False").lower() == "true"
    
    # Additional API keys and services
    e2b_api_key: Optional[str] = None
    serpapi_api_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    
    # Allow any extra fields in case we add more API keys in the future
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Allow extra fields
    }

def get_settings() -> Settings:
    """
    Get the application settings.
    
    Returns:
        Settings: The application settings
    """
    return Settings()
