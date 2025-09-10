# runners.py
from __future__ import annotations
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING
from ..config import Config

if TYPE_CHECKING:
    from src.moshousapient.components.camera_worker import CameraWorker

class BaseRunner(ABC):
    """
    執行策略的抽象基礎類別。
    定義了所有具體執行器 (Runner) 的共同介面與生命週期管理。
    """

    def __init__(self, workers: List[CameraWorker], notifier):
        self.workers = workers
        self.notifier = notifier
        self.stop_event = threading.Event()

    def start_workers(self):
        """啟動所有已設定的 CameraWorker。"""
        for worker in self.workers:
            worker.start()

    @abstractmethod
    def run(self):
        """
        【抽象方法】
        啟動並執行主要的監控邏輯。
        每個子類別必須實作自己獨特的監控迴圈。
        """
        pass

    def shutdown(self):
        """
        執行一個統一、優雅的關閉程序。
        這個方法確保了所有 worker 和 notifier 都有機會完成它們的工作。
        """
        logging.info("[系統] 正在優雅地關閉所有服務, 請稍候...")

        for worker in self.workers:
            worker.stop()

        if self.notifier:
            self.notifier.stop()

        logging.info("[系統] 系統已安全關閉。")


class RTSPRunner(BaseRunner):
    """
    針對 RTSP 即時串流的執行策略。
    此模式下，系統會永久運行，直到手動中斷或發生嚴重錯誤。
    """

    def run(self):
        logging.info("[系統] 進入 RTSP (永久監控) 模式。")
        self.start_workers()

        while not self.stop_event.is_set():
            time.sleep(Config.HEALTH_CHECK_INTERVAL)
            for worker in self.workers:
                if not worker.is_alive():
                    logging.critical(f"[{worker.name}] 的一個或多個核心執行緒已停止運作! 系統將準備關閉。")
                    self.stop_event.set()
                    break


class FileRunner(BaseRunner):
    """
    針對本地檔案處理的執行策略。
    此模式下，系統會在所有影格處理完畢後自動、優雅地關閉。
    """

    def run(self):
        logging.info(f"[系統] 進入 FILE (批次處理) 模式。等待所有 worker 完成...")
        self.start_workers()

        while not self.stop_event.is_set():
            if not any(w.is_alive() for w in self.workers):
                logging.info("[系統] 所有 worker 均已完成其主要任務。觸發系統最終關閉程序。")
                self.stop_event.set()
            else:
                time.sleep(1)
