import logging
import os
from datetime import datetime

# Log directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log filename with timestamp
log_filename = os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for a module.
    
    Usage in any module:
        from app.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started")
    
    Args:
        name: Module name — use __name__ always
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Format — timestamp, level, module name, message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler — INFO and above shown in terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # File handler — DEBUG and above saved to log file
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger