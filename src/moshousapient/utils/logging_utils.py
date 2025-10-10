# src/moshousapient/utils/logging_utils.py
"""
提供與日誌系統相關的進階輔助工具。
"""

import logging
import sys


class StreamToLogger:
    """
    一個偽檔案流 (pseudo-file stream)，可將寫入其中的數據重定向到指定的 logger。
    用於捕獲標準輸出 (stdout) 和標準錯誤 (stderr)。
    """

    def __init__(self, logger: logging.Logger, level: int):
        """
        初始化 StreamToLogger。

        :param logger: 目標 logger 實例。
        :param level: 重定向訊息時使用的日誌級別 (例如 logging.DEBUG)。
        """
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf: str):
        """
        實現 file-like 的 write 方法。
        """
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        """
        實現 file-like 的 flush 方法。
        """
        pass

    def __enter__(self):
        """上下文管理器進入方法：保存並重定向標準流。"""
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出方法：恢復原始標準流。"""
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
