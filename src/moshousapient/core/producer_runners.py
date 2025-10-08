# src/moshousapient/core/producer_runners.py
"""
此模組定義了應用程式執行的策略模式，包含了不同影像來源的執行器 (Runner) 類別。

- BaseRunner: 所有執行策略的抽象基礎類別。
- RTSPProducerRunner: 負責管理和監控一個或多個即時 RTSP 處理管線。
"""

# 1. 標準庫導入
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Any

# 2. 第三方庫導入
# (無)

# 3. 本專案相對導入
from ..configs.behavior_config import Config


class BaseRunner(ABC):
    """執行策略的抽象基礎類別。"""

    def __init__(self, workers: List[Any], notifier):
        """
        初始化基礎 Runner。

        :param workers: 由此 Runner 管理的 Worker/Pipeline 物件列表。
        :param notifier: 用於發送通知的服務實例。
        """
        self.workers = workers
        self.notifier = notifier

    def start_workers(self):
        """啟動所有關聯的 Worker。"""
        for worker in self.workers:
            worker.start()

    @abstractmethod
    def run(self):
        """【抽象方法】執行 Runner 的主要邏輯。"""
        pass

    def shutdown(self):
        """停止所有關聯的 Worker。"""
        logging.debug("[系統] 正在關閉 Runner 相關服務...")
        if self.workers:
            for worker in self.workers:
                worker.stop()


class RTSPProducerRunner(BaseRunner):
    """針對 RTSP 即時串流的執行策略。"""

    def run(self):
        """
        啟動並持續監控 RTSP 處理管線的健康狀態。
        """
        self.start_workers()
        time.sleep(5)  # 給予管線啟動和緩衝時間

        if not all(w.is_alive() for w in self.workers):
            logging.critical("一個或多個處理管線未能成功啟動，系統將關閉。")
            return

        logging.info("所有 RTSP 處理管線已成功啟動並運行中。")
        try:
            while True:
                time.sleep(Config.HEALTH_CHECK_INTERVAL)
                if not all(w.is_alive() for w in self.workers):
                    logging.critical("偵測到處理管線異常停止！系統將準備關閉。")
                    break
        except KeyboardInterrupt:
            # 允許透過 Ctrl+C 優雅退出
            pass