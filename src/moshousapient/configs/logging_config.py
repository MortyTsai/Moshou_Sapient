# src/moshousapient/configs/logging_config.py
"""
此模組提供基於佇列的中央化日誌設定功能，實現使用者日誌與開發者日誌的分離。

- 使用者日誌 (INFO 級別及以上): 輸出到主控台，格式簡潔。
- 開發者日誌 (DEBUG 級別及以上): 輸出到日誌檔案，格式詳細，用於除錯和追蹤。
"""

import logging
import logging.handlers
import sys
import os
from queue import Queue

from .settings_config import settings


class LevelFilter(logging.Filter):
    """
    根據指定的最低日誌級別過濾日誌記錄。
    """
    def __init__(self, low_level):
        super().__init__()
        self.low_level = low_level

    def filter(self, record):
        return record.levelno >= self.low_level


def configure_logging_for_queue(log_queue: Queue):
    """
    為當前程序配置日誌系統，使其將所有日誌發送到指定的佇列。

    :param log_queue: 所有日誌記錄將被放入此佇列。
    """
    root_logger = logging.getLogger()
    # 捕獲所有級別的日誌，過濾交給監聽器
    root_logger.setLevel(logging.DEBUG)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    queue_handler = logging.handlers.QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)

    # 為特定函式庫設定較高的日誌級別以減少干擾
    logging.getLogger("BehaviorAnalysis").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("discord").setLevel(logging.ERROR)


def setup_logging_listener(log_queue: Queue):
    """
    設定並返回一個日誌監聽器。
    該監聽器從佇列中獲取日誌，並根據規則分發到主控台和檔案。

    :param log_queue: 監聽器將從此佇列中獲取日誌。
    :return: 配置好但尚未啟動的 logging.handlers.QueueListener 物件。
    """
    # --- 1. 建立不同的日誌格式 ---
    dev_formatter = logging.Formatter(
        '%(asctime)s - PID:%(process)-6d - %(threadName)-25s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    user_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s', datefmt='%H:%M:%S')

    # --- 2. 設定檔案處理器 (開發者日誌) ---
    log_dir = settings.DATA_DIR / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = log_dir / "moshousapient.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setFormatter(dev_formatter)
    # 檔案處理器不過濾，記錄所有 DEBUG 及以上級別的日誌
    file_handler.setLevel(logging.DEBUG)

    # --- 3. 設定主控台處理器 (使用者日誌) ---
    console_log_level_str = settings.LOG_LEVEL.upper()
    console_log_level = getattr(logging, console_log_level_str, logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(user_formatter)
    console_handler.addFilter(LevelFilter(console_log_level))

    # --- 4. 建立並返回監聽器 ---
    listener = logging.handlers.QueueListener(log_queue, file_handler, console_handler, respect_handler_level=True)
    return listener