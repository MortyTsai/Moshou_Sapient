# src/moshousapient/core/worker_manager.py
"""
此模組提供 WorkerManager，用於管理背景工作程序 (Consumers) 的生命週期。
它負責根據應用程式設定啟動、監控和優雅地關閉 Worker 程序池。
"""

import logging
import multiprocessing as mp
import time
from typing import List

from ..services.task_queue_service import TaskQueueService
from ..workers.video_consumer_worker import worker_entrypoint


class WorkerManager:
    """
    管理 VideoConsumerWorker 子程序池的生命週期。
    """

    def __init__(self, num_workers: int, log_queue: mp.Queue):
        """
        初始化 Worker 管理器。

        :param num_workers: 要啟動和管理的 Worker 程序的數量。
        :param log_queue: 用於中央化日誌的 multiprocessing 佇列。
        """
        if num_workers < 1:
            raise ValueError("Worker 數量必須至少為 1。")
        self.num_workers = num_workers
        self.log_queue = log_queue
        self.worker_processes: List[mp.Process] = []
        self.task_queue = TaskQueueService()
        logging.debug(f"[WorkerManager] 已初始化，將管理 {self.num_workers} 個 Worker。")

    def _cleanup_stale_tasks(self):
        """清理上次運行可能遺留下來的過期任務。"""
        logging.debug("[WorkerManager] 正在檢查是否有因上次異常關閉而遺留的任務需要清理...")
        deleted_count = self.task_queue.delete_stale_tasks(timeout_seconds=600)
        if deleted_count > 0:
            logging.warning(f"[WorkerManager] 已成功清理 {deleted_count} 個過期任務。")
        else:
            logging.debug("[WorkerManager] 任務佇列狀態乾淨，無需清理。")

    def start_workers(self):
        """
        啟動所有背景 Worker 程序。
        """
        self._cleanup_stale_tasks()
        logging.info(f"[WorkerManager] 正在啟動 {self.num_workers} 個背景處理程序...")

        for i in range(self.num_workers):
            try:
                process = mp.Process(
                    target=worker_entrypoint,
                    args=(self.log_queue,),
                    name=f"Worker-{i + 1}",
                )
                process.daemon = True
                self.worker_processes.append(process)
                process.start()
                logging.debug(f"[WorkerManager] {process.name} (PID: {process.pid}) 已成功啟動。")
            except Exception as e:
                logging.critical(f"[WorkerManager] 啟動 Worker #{i + 1} 失敗: {e}", exc_info=True)
                self.shutdown_workers()
                raise

    def shutdown_workers(self):
        """
        向所有 Worker 程序發送優雅關閉信號，並等待它們終止。
        """
        if not self.worker_processes:
            return

        logging.debug(f"[WorkerManager] 正在向 {len(self.worker_processes)} 個 Worker 發送關閉信號...")
        for process in self.worker_processes:
            if process.is_alive():
                try:
                    process.terminate()
                except Exception as e:
                    logging.error(f"發送終止信號到 {process.name} 失敗: {e}")

        logging.debug("[WorkerManager] 等待所有 Worker 終止...")
        timeout = 10
        start_time = time.time()

        remaining_processes = list(self.worker_processes)
        while remaining_processes and (time.time() - start_time) < timeout:
            remaining_processes = [p for p in remaining_processes if p.is_alive()]
            time.sleep(0.1)

        for process in remaining_processes:
            if process.is_alive():
                logging.warning(f"[WorkerManager] {process.name} 優雅關閉超時，將強制終止。")
                process.kill()

        self.worker_processes.clear()
        logging.debug("[WorkerManager] 所有 Worker 已成功關閉。")
