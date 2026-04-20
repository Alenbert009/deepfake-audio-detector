import logging
import os
from app.config import LOG_PATH
def setup_logger():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logger = logging.getLogger("app_logger")

    if not logger.handlers:  # 🔥 prevents duplicate handlers
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger