# src/moshousapient/core/producer_runners.py
"""
此模組定義了應用程式執行的策略模式，包含了不同影像來源的執行器 (Runner) 類別。
"""

import logging
import time
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any, Optional
import threading
import sys
import subprocess

from ..configs.behavior_config import Config
from ..processors.file_event_producer import FileEventProducer
from ..configs.settings_config import PROJECT_ROOT
from ..services.task_queue_service import TaskQueueService
from ..services.notification_service import NotificationService


class BaseRunner(ABC):
    """執行策略的抽象基礎類別。"""

    def __init__(self, workers: List[Any], notifier):
        self.workers = workers
        self.notifier = notifier

    def start_workers(self):
        for worker in self.workers:
            worker.start()

    @abstractmethod
    def run(self):
        pass

    def shutdown(self):
        logging.debug("[系統] 正在關閉 Runner 相關服務...")
        if self.workers:
            for worker in self.workers:
                worker.stop()


class RTSPProducerRunner(BaseRunner):
    """針對 RTSP 即時串流的執行策略。"""

    def run(self):
        self.start_workers()
        time.sleep(5)
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
            pass


class FileProducerRunner(BaseRunner):
    """針對本地檔案處理的執行策略 (v6 - subprocess + thread redirect)。"""

    def __init__(self, workers: List[Any], notifier):
        super().__init__(workers, notifier)
        self.result_processor = FileEventProducer(notifier)

    def run(self):
        """執行基於子程序的檔案處理流程，並從父程序即時捕獲其輸出。"""
        video_path_str = Config.VIDEO_FILE_PATH
        if not video_path_str:
            logging.error("錯誤: 在 FILE 模式下未設定 VIDEO_FILE_PATH。")
            return
        video_path = Path(video_path_str)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        if not video_path.exists():
            logging.error(f"錯誤: 影片檔案不存在: {video_path}")
            return

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json', encoding='utf-8') as temp_f:
            json_output_path = Path(temp_f.name)

        process = None
        try:
            logging.info(f"正在啟動隔離程序對影片 '{video_path.name}' 進行分析...")

            command = [
                sys.executable, "-m", "moshousapient.services.isolated_inference_service",
                "--video-path", str(video_path.resolve()),
                "--output-json-path", str(json_output_path.resolve()),
                "--behavior-config-path", str(Config.BEHAVIOR_CONFIG_PATH)
            ]

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )

            stdout_thread = threading.Thread(target=self._log_stream, args=(process.stdout, "IsoInference-out"))
            stderr_thread = threading.Thread(target=self._log_stream, args=(process.stderr, "IsoInference-err"))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            process.wait()
            stdout_thread.join()
            stderr_thread.join()

            if process.returncode != 0:
                logging.error(f"隔離推論程序執行失敗，返回碼: {process.returncode}")
                return

            logging.debug("隔離推論程序執行成功。正在讀取 JSON 結果...")
            with open(json_output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

            if results:
                self.result_processor.process_results(results)

            logging.info("檔案分析與任務分派完成。等待背景處理程序完成...")

            task_queue_monitor = TaskQueueService()
            notifier_monitor: Optional[NotificationService] = self.notifier

            while True:
                time.sleep(2)

                tasks_left = task_queue_monitor.get_pending_or_processing_count()
                notifications_left = notifier_monitor.get_pending_tasks_count() if notifier_monitor else 0

                logging.debug(f"監控中... 待處理任務: {tasks_left}, 待發送通知: {notifications_left}")

                if tasks_left == 0 and notifications_left == 0:
                    time.sleep(2)
                    logging.info("所有任務已處理完成。您可以隨時按 Ctrl+C 來終止程序。")
                    break

            while True:
                time.sleep(10)

        except (Exception, KeyboardInterrupt):
            pass
        finally:
            if process and process.poll() is None:
                process.kill()
            if json_output_path and json_output_path.exists():
                json_output_path.unlink()
            logging.debug("檔案處理流程結束。")

    @staticmethod
    def _log_stream(stream, logger_name: str):
        """讀取流的內容並將其作為 DEBUG 日誌發送。"""
        logger = logging.getLogger(logger_name)
        try:
            for line in iter(stream.readline, ''):
                if line:
                    logger.debug(line.strip())
        except (IOError, ValueError):
            pass