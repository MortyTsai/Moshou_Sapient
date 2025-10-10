# src/moshousapient/processors/base_processor.py
"""
定義了所有處理器 (Processor) 的抽象基礎類別。
"""

# 1. 標準庫導入
import threading
from abc import ABC, abstractmethod
from typing import Optional


class BaseProcessor(ABC):
    """
    一個抽象基礎類別，定義了所有處理器共有的介面和生命週期管理。

    每個處理器都在一個獨立的執行緒中運行，並可以被安全地啟動和停止。
    """

    def __init__(self, name: str):
        """
        初始化基礎處理器。

        :param name: 處理器的名稱，將用作執行緒名稱。
        """
        self.name = name
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    @abstractmethod
    def _target_func(self):
        """
        【抽象方法】執行緒將執行的主要目標函式。
        子類必須實現此方法來定義其核心處理 logique。
        """
        pass

    def start(self):
        """
        啟動處理器執行緒。
        如果執行緒已經在運行，此操作將被忽略。
        """
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._target_func, name=self.name)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        """
        向處理器發送停止信號，並等待執行緒終止。
        """
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def is_alive(self) -> bool:
        """
        檢查處理器執行緒是否仍在活動。

        :return: 如果執行緒存在且正在運行，則返回 True。
        """
        return self.thread is not None and self.thread.is_alive()
