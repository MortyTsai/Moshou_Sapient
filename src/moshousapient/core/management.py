# src/moshousapient/core/management.py
"""
此模組提供 WorkerManager，用於管理背景工作程序（Consumers）的生命週期。
它負責根據應用程式設定啟動、監控和優雅地關閉 Worker 程序池。
"""

# 1. 標準庫導入
import logging
import subprocess
import sys
import signal
import time
import os
from typing import List

# 3. 本專案相對導入
from ..config import Config
from ..services.task_queue_service import TaskQueueService


class WorkerManager:
    """
    管理 VideoProcessingWorker 子程序池的生命週期。
    """

    def __init__(self, num_workers: int):
        """
        初始化 Worker 管理器。

        :param num_workers: 要啟動和管理的 Worker 程序的數量。
        """
        if num_workers < 1:
            raise ValueError("Worker 數量必須至少為 1。")
        self.num_workers = num_workers
        self.worker_processes: List[subprocess.Popen] = []
        self.task_queue = TaskQueueService()
        logging.info(f"[WorkerManager] 已初始化，將管理 {self.num_workers} 個 Worker。")

    def _cleanup_stale_tasks(self):
        """
        清理上次運行可能遺留下來的過期任務。
        """
        logging.info("[WorkerManager] 正在檢查是否有過期的任務需要重新排入佇列...")
        # 使用一個較長的超時時間，例如 10 分鐘，以處理上次運行崩潰的任務
        self.task_queue.requeue_stale_tasks(timeout_seconds=600)

    def start_workers(self):
        """
        啟動所有背景 Worker 程序。
        在啟動前，會先清理可能存在的過期任務。
        """
        self._cleanup_stale_tasks()

        logging.info(f"[WorkerManager] 正在啟動 {self.num_workers} 個 VideoProcessingWorker...")
        command = [
            sys.executable,
            "-m",
            "moshousapient.workers.video_processing_worker"
        ]

        for i in range(self.num_workers):
            try:
                process = subprocess.Popen(
                    command,
                    stdout=sys.stdout,
                    stderr=sys.stderr
                )
                self.worker_processes.append(process)
                logging.info(f"[WorkerManager] Worker #{i + 1} (PID: {process.pid}) 已成功啟動。")
            except (OSError, FileNotFoundError) as e:
                logging.critical(f"[WorkerManager] 啟動 Worker #{i + 1} 失敗: {e}", exc_info=True)
                self.shutdown_workers()
                raise

    def shutdown_workers(self):
        """
        向所有 Worker 程序發送優雅關閉信號，並等待它們終止。
        此方法已針對 Windows 和 Unix-like 系統進行了兼容性處理。
        """
        if not self.worker_processes:
            return

        logging.info(f"[WorkerManager] 正在向 {len(self.worker_processes)} 個 Worker 發送關閉信號...")

        for process in self.worker_processes:
            if process.poll() is None:
                try:
                    if os.name == 'nt':  # Windows
                        process.send_signal(signal.CTRL_C_EVENT)
                    else:  # Unix-like
                        process.send_signal(signal.SIGINT)
                except (OSError, ValueError):
                    try:
                        process.terminate()
                    except OSError:
                        pass

        timeout = Config.THREAD_JOIN_TIMEOUT
        logging.info(f"[WorkerManager] 等待所有 Worker 終止 (超時: {timeout} 秒)...")

        start_time = time.time()
        remaining_processes = list(self.worker_processes)

        while remaining_processes and (time.time() - start_time) < timeout:
            remaining_processes = [p for p in remaining_processes if p.poll() is None]
            time.sleep(0.5)

        if remaining_processes:
            logging.warning("[WorkerManager] 優雅關閉超時，將強制終止剩餘的 Worker。")
            for process in remaining_processes:
                if process.poll() is None:
                    logging.warning(f"[WorkerManager] 強制終止 Worker (PID: {process.pid})。")
                    process.kill()

        self.worker_processes.clear()
        logging.info("[WorkerManager] 所有 Worker 已成功關閉。")