# src/moshousapient/core/producer_runners.py
"""
此模組定義了應用程式執行的策略模式，包含了不同影像來源的執行器 (Runner) 類別。

Runner 負責啟動並管理特定模式下的生產者 (Producer) 流程。
"""

# 1. 標準庫導入
import logging
import subprocess
import sys
import tempfile
import time
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any

# 3. 本專案相對導入
from ..configs.behavior_config import Config
from ..processors.file_event_producer import FileEventProducer
from ..configs.settings_config import PROJECT_ROOT


class BaseRunner(ABC):
    """
    執行策略的抽象基礎類別。
    """

    def __init__(self, workers: List[Any], notifier):
        """
        初始化基礎執行器。

        :param workers: (在此次重構中已部分廢棄) 工作單元列表。
        :param notifier: 用於發送通知的通知器物件。
        """
        self.workers = workers
        self.notifier = notifier

    def start_workers(self):
        """啟動所有已設定的 Worker。"""
        for worker in self.workers:
            worker.start()

    @abstractmethod
    def run(self):
        """【抽象方法】啟動並執行主要的監控邏輯。"""
        pass

    def shutdown(self):
        """執行一個統一、優雅的關閉程序。"""
        logging.info("[系統] 正在優雅地關閉所有服務，請稍候...")
        if self.workers:
            for worker in self.workers:
                worker.stop()
        if self.notifier:
            self.notifier.stop()
        logging.info("[系統] 系統已安全關閉。")


class RTSPProducerRunner(BaseRunner):
    """
    針對 RTSP 即時串流的執行策略。
    負責管理一個或多個 RTSP 處理管線 (Pipeline) 的生命週期。
    """

    def run(self):
        """啟動永久監控模式，並透過健康檢查來監控 Worker 狀態。"""
        logging.info("[系統] 進入 RTSP (永久監控) 模式。")
        self.start_workers()
        time.sleep(5)  # 給予 Worker 啟動時間

        if not all(w.is_alive() for w in self.workers):
            logging.critical("[系統] 一個或多個處理管線未能成功啟動，系統將關閉。")
            return

        logging.info("[系統] 所有處理管線已成功啟動並運行中。")
        try:
            while True:
                time.sleep(Config.HEALTH_CHECK_INTERVAL)
                if not all(w.is_alive() for w in self.workers):
                    logging.critical("[系統] 偵測到處理管線異常停止！系統將準備關閉。")
                    break
        except KeyboardInterrupt:
            logging.info("\n[系統] 收到關閉信號 (Ctrl+C)...")


class FileProducerRunner(BaseRunner):
    """
    針對本地檔案處理的執行策略 (v2 - 任務佇列版)。

    此執行器透過 subprocess 呼叫推論服務，然後將結果分派到任務佇列，
    並等待處理完成或手動中斷。
    """

    def __init__(self, workers: List[Any], notifier):
        """
        初始化檔案執行器，並創建一個結果處理器。
        """
        super().__init__(workers, notifier)
        self.result_processor = FileEventProducer(notifier)

    def run(self):
        """執行基於子程序的檔案處理流程。"""
        logging.info("[FileRunner] 進入 FILE (隔離程序) 模式。")
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

        json_output_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json', encoding='utf-8') as temp_f:
                json_output_path = Path(temp_f.name)

            command = [
                sys.executable, "-m", "moshousapient.services.isolated_inference_service",
                "--video-path", str(video_path.resolve()),
                "--output-json-path", str(json_output_path.resolve()),
                "--behavior-config-path", str(Config.BEHAVIOR_CONFIG_PATH)
            ]

            logging.info(f"[FileRunner] 準備執行子程序，結果將輸出至 {json_output_path}")
            process = subprocess.run(
                command, capture_output=True, text=True,
                encoding='utf-8', check=False
            )

            if process.stdout:
                logging.info(f"[FileRunner] 子程序 STDOUT:\n--- START ---\n{process.stdout.strip()}\n--- END ---")
            if process.stderr:
                logging.warning(f"[FileRunner] 子程序 STDERR:\n--- START ---\n{process.stderr.strip()}\n--- END ---")

            if process.returncode != 0:
                logging.error(f"[FileRunner] 子程序執行失敗，返回碼: {process.returncode}")
                return

            logging.info("[FileRunner] 子程序執行成功。正在讀取 JSON 結果...")
            with open(json_output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

            if results:
                self.result_processor.process_results(results)

            logging.info("[FileRunner] 檔案分析與任務分派完成。主程序將保持運行以等待背景 Worker 處理任務。")
            logging.info("您可以隨時按 Ctrl+C 來終止所有程序。")
            while True:
                time.sleep(10)

        except (Exception, KeyboardInterrupt) as e:
            if not isinstance(e, KeyboardInterrupt):
                logging.critical(f"[FileRunner] 執行子程序時發生未預期的錯誤: {e}", exc_info=True)
        finally:
            if json_output_path and json_output_path.exists():
                json_output_path.unlink()
            logging.info("[FileRunner] 檔案處理流程結束。")