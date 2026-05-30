import os
import yaml
from app.logger import get_logger

logger = get_logger(__name__)

# Path to config file
CONFIG_PATH = "config.yaml"

# Singleton — config loaded once
_config = None


def get_config() -> dict:
    """
    Load configuration from config.yaml — singleton pattern.
    Config loaded once and reused for all subsequent calls.
    
    Returns:
        dict: Complete configuration
    """
    global _config
    if _config is None:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
        
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {CONFIG_PATH}")
    
    return _config


def get_pipeline_config() -> dict:
    """Get pipeline configuration section."""
    return get_config()["pipeline"]


def get_storage_config() -> dict:
    """Get storage paths configuration section."""
    return get_config()["storage"]


def get_rag_config() -> dict:
    """Get RAG configuration section."""
    return get_config()["rag"]