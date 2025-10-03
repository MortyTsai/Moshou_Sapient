# src/moshousapient/logging_setup.py
"""
此模組提供全域的日誌設定功能。
"""

# 1. 標準庫導入
import logging
import sys


def setup_logging():
    """
    設定全域的 logging，包含格式與輸出位置（主控台）。

    新的格式包含了進程 ID (PID)，以便在多程序環境下（如 Worker 池）
    清晰地區分日誌來源。
    """
    # 格式: 時間 - 進程ID - 執行緒名稱 - 日誌級別 - 訊息
    log_formatter = logging.Formatter(
        '%(asctime)s - PID:%(process)-6d - %(threadName)-25s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()

    # 避免在重複導入時添加多個 handler
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.INFO)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    root_logger.addHandler(stdout_handler)

    # 為特定的 logger 設定不同的級別，以減少干擾
    behavior_logger = logging.getLogger("BehaviorAnalysis")
    behavior_logger.setLevel(logging.WARNING)

    logging.info("日誌系統已成功初始化。")