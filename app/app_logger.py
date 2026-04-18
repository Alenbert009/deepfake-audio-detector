import logging
from app.config import LOG_PATH
def setup_logger():
    logger=logging.getLogger("DeepFakeAudioDetector")
    logger.setLevel(logging.INFO)
    file_handler=logging.FileHandler(LOG_PATH)
    formatter=logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
        
    return logger