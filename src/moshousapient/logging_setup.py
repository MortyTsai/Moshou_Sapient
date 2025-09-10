# logging_setup.py
import logging
import sys

def setup_logging():
    """
    設定全域的 logging, 包含格式與輸出位置 (主控台)。
    """
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
    logging.info("日誌系統已成功初始化。")
