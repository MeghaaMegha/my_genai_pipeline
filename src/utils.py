import logging
import sys


def get_production_logger(logger_name: str) -> logging.Logger:
    """
    Get a logger configured for production use.

    Args:
        logger_name (str): The name of the logger.
    Returns:
        logging.Logger: A logger instance configured for production.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs if handlers are already set up
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] - %(message)s"
        )

        # Stream to terminal output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger
