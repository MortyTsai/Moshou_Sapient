# logging_setup.py
import logging
import sys
from logging.handlers import TimedRotatingFileHandler


def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s - %(threadName)-25s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    root_logger.addHandler(stdout_handler)

    file_handler = TimedRotatingFileHandler("app.log", when="midnight", interval=1, backupCount=7)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    logging.info("日誌系統已成功初始化。")