# src/moshousapient/core/runners.py

import logging
import threading
import time
import subprocess
import sys
import tempfile
import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Any

from ..config import Config
from ..settings import PROJECT_ROOT
from ..processors.file_result_processor import FileResultProcessor


class BaseRunner(ABC):
    """執行策略的抽象基礎類別。"""

    def __init__(self, workers: List[Any], notifier):
        """初始化基礎執行器。"""
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
        logging.info("[系統] 正在優雅地關閉所有服務, 請稍候...")
        if self.workers:
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

        time.sleep(5)
        if not all(w.is_alive() for w in self.workers):
            logging.critical("[系統] 一個或多個 Worker 未能成功啟動, 系統將關閉。")
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
    """
    針對本地檔案處理的執行策略 (v8.1.1 架構重塑版)。
    此執行器透過 subprocess 呼叫一個獨立的推論服務來處理影片，
    實現了程序級隔離。
    """

    def __init__(self, workers: List[Any], notifier):
        super().__init__(workers, notifier)
        self.result_processor = FileResultProcessor(notifier)

    def run(self):
        logging.info(f"[FileRunner] 進入 FILE (隔離程序) 模式。")
        video_path_str = Config.VIDEO_FILE_PATH
        if not video_path_str:
            logging.critical("[FileRunner] 錯誤: 在 FILE 模式下未設定 VIDEO_FILE_PATH。")
            return

        video_path = Path(video_path_str)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        if not video_path.exists():
            logging.critical(f"[FileRunner] 錯誤: 影片檔案不存在: {video_path}")
            return

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json', encoding='utf-8') as temp_output_file:
            json_output_path = Path(temp_output_file.name)

        command = [
            sys.executable, "-m", "moshousapient.services.isolated_inference_service",
            "--video-path", str(video_path.resolve()),
            "--output-json-path", str(json_output_path.resolve()),
            "--behavior-config-path", str(Config.BEHAVIOR_CONFIG_PATH)
        ]
        logging.info(f"[FileRunner] 準備執行子程序，結果將輸出至 {json_output_path}")

        try:
            process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False)
            if process.stdout: logging.info(
                f"[FileRunner] 子程序 STDOUT:\n--- START ---\n{process.stdout.strip()}\n--- END ---")
            if process.stderr: logging.warning(
                f"[FileRunner] 子程序 STDERR:\n--- START --- \n{process.stderr.strip()}\n--- END ---")
            if process.returncode != 0:
                logging.error(f"[FileRunner] 子程序執行失敗，返回碼: {process.returncode}")
                return

            logging.info(f"[FileRunner] 子程序執行成功。正在讀取 JSON 結果...")
            with open(json_output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

            if results:
                self.result_processor.process_results(results)
        except Exception as e:
            logging.critical(f"[FileRunner] 執行子程序時發生未預期的錯誤: {e}", exc_info=True)
        finally:
            if json_output_path.exists():
                json_output_path.unlink()
            logging.info("[FileRunner] 檔案處理流程結束。")