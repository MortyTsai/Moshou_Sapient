# src/moshousapient/core/runners.py

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Any

from ..config import Config


class BaseRunner(ABC):
    """執行策略的抽象基礎類別。"""

    def __init__(self, workers: List[Any], notifier):
        """
        初始化基礎執行器。

        :param workers: 一個 CameraWorker 物件的列表。
        :param notifier: 一個通知器實例。
        """
        self.workers = workers
        self.notifier = notifier
        self.stop_event = threading.Event()

    def start_workers(self):
        """啟動所有已設定的 CameraWorker。"""
        for worker in self.workers:
            worker.start()

    @abstractmethod
    def run(self):
        """【抽象方法】啟動並執行主要的監控邏輯。"""
        pass

    def shutdown(self):
        """執行一個統一、優雅的關閉程序。"""
        logging.info("[系統] 正在優雅地關閉所有服務，請稍候...")
        for worker in self.workers:
            worker.stop()
        if self.notifier:
            self.notifier.stop()
        logging.info("[系統] 系統已安全關閉。")


class RTSPRunner(BaseRunner):
    """針對 RTSP 即時串流的執行策略。"""

    def run(self):
        logging.info("[系統] 進入 RTSP (永久監控) 模式。")
        self.start_workers()

        # 初始健康檢查，確保所有元件在啟動時都正常
        time.sleep(5)
        if not all(w.is_alive() for w in self.workers):
            logging.critical("[系統] 一個或多個 Worker 未能成功啟動，系統將關閉。")
            self.stop_event.set()
            return

        logging.info("[系統] 所有 Worker 已成功啟動並運行中。")
        while not self.stop_event.is_set():
            time.sleep(Config.HEALTH_CHECK_INTERVAL)
            if not all(w.is_alive() for w in self.workers):
                logging.critical(f"[系統] 偵測到 Worker 異常停止! 系統將準備關閉。")
                self.stop_event.set()
                break


class FileRunner(BaseRunner):
    """針對本地檔案處理的執行策略。"""

    def run(self):
        logging.info(f"[系統] 進入 FILE (批次處理) 模式。")
        self.start_workers()

        # 等待，直到所有 worker 的 VideoStreamer 完成它的工作
        # worker.is_alive() 只檢查 streamer 的狀態
        for worker in self.workers:
            while worker.is_alive():
                time.sleep(1)

        logging.info("[系統] 影片來源處理完畢。給予下游處理器寬限期以完成剩餘任務...")

        # 給予一個固定的寬限期，確保佇列中的剩餘幀能被完全處理
        grace_period = Config.POST_EVENT_SECONDS + 5.0  # 比後期緩衝稍長的時間
        time.sleep(grace_period)

        logging.info("[系統] 所有任務已完成。觸發系統最終關閉程序。")