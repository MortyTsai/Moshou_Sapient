# src/moshousapient/core/scheduler.py
"""
智慧排程器，負責管理資源密集型的背景任務。

此排程器現在是一個純事件驅動的元件。它監控 RTSP 的事件活躍狀態
和任務佇列，以決定是否啟動背景 Worker。

其核心目標是確保高優先級的即時任務（如 RTSP 分析）的資源不受影響，
僅在系統閒置時執行低優先級的任務，並能在即時事件發生時立即搶佔資源。
"""

# 1. 標準庫導入
import logging
import subprocess
import sys
import threading
from typing import Optional, Any

# 2. 第三方庫導入
# (無)

# 3. 本專案相對導入
from ..configs.settings_config import settings
from ..services.task_queue_service import TaskQueueService


class Scheduler:
    """
    一個常駐的背景服務，用於根據系統負載動態啟動任務處理程序。
    """

    def __init__(self, task_queue: TaskQueueService, rtsp_event_active_flag: Any):
        """
        初始化智慧排程器。

        :param task_queue: 用於檢查待辦任務的任務佇列服務實例。
        :param rtsp_event_active_flag: 一個跨程序共享的布林標誌，指示 RTSP 事件是否活躍。
        """
        self._task_queue = task_queue
        self._rtsp_event_active_flag = rtsp_event_active_flag
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._worker_process: Optional[subprocess.Popen] = None

        self._check_interval_seconds = settings.SCHEDULER_CHECK_INTERVAL

        logging.debug(f"[Scheduler] 已初始化。檢查間隔: {self._check_interval_seconds}s")

    def _terminate_worker_process(self, reason: str):
        """
        安全地終止正在運行的 Worker 子程序。

        :param reason: 終止的原因，用於日誌記錄。
        """
        if not self._worker_process or self._worker_process.poll() is not None:
            return

        logging.warning(f"[Scheduler] 正在終止佇列推論 Job (PID: {self._worker_process.pid})，原因: {reason}")
        try:
            self._worker_process.terminate()
            self._worker_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logging.error(f"[Scheduler] Job (PID: {self._worker_process.pid}) 未能優雅終止，將強制終止。")
            self._worker_process.kill()
        finally:
            self._worker_process = None

    def _launch_worker(self):
        """
        啟動一個新的 queue_inference_job 子程序。
        """
        try:
            command = [sys.executable, "-m", "moshousapient.jobs.queue_inference_job"]
            self._worker_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info(f"[Scheduler] 已成功啟動佇列推論 Job，PID: {self._worker_process.pid}")
        except Exception as e:
            logging.critical(f"[Scheduler] 啟動佇列推論 Job 失敗: {e}", exc_info=True)
            self._worker_process = None

    def _run(self):
        """
        排程器的主循環，在背景執行緒中運行。
        """
        logging.info("[Scheduler] 事件驅動排程器已啟動。")

        while not self._stop_event.is_set():
            try:
                is_rtsp_active = self._rtsp_event_active_flag.value

                # 1. 搶佔式檢查
                if is_rtsp_active and self._worker_process and self._worker_process.poll() is None:
                    self._terminate_worker_process("RTSP 事件觸發，搶佔資源")

                # 2. 檢查已完成的 Worker
                if self._worker_process and self._worker_process.poll() is not None:
                    logging.info(f"[Scheduler] 佇列推論 Job (PID: {self._worker_process.pid}) 已完成工作。")
                    self._worker_process = None

                # 3. 檢查是否滿足啟動新 Worker 的條件
                if self._worker_process is None:
                    rescued_count = self._task_queue.reset_stale_processing_tasks(timeout_seconds=10)
                    if rescued_count > 0:
                        logging.info(f"[Scheduler] 已成功救援 {rescued_count} 個卡住的任務。")

                    has_pending_tasks = self._task_queue.has_pending_task_by_type("file_inference")

                    if has_pending_tasks and not is_rtsp_active:
                        logging.info("[Scheduler] 偵測到待辦任務且 RTSP 閒置，準備啟動佇列推論 Job。")
                        self._launch_worker()
                    else:
                        reasons = []
                        if not has_pending_tasks:
                            reasons.append("無待辦任務")
                        if is_rtsp_active:
                            reasons.append("RTSP 事件活躍")
                        if reasons:
                            logging.debug(f"[Scheduler] 未滿足啟動條件 ({', '.join(reasons)})。")
                else:
                    logging.debug(f"[Scheduler] 佇列推論 Job (PID: {self._worker_process.pid}) 正在運行中。")

            except Exception as e:
                logging.error(f"[Scheduler] 主循環發生錯誤: {e}", exc_info=True)

            self._stop_event.wait(self._check_interval_seconds)

        # 清理工作
        self._terminate_worker_process("排程器正在關閉")
        logging.info("[Scheduler] 已關閉。")

    def start(self):
        """在一個獨立的背景執行緒中啟動排程器服務。"""
        if self._thread and self._thread.is_alive():
            logging.warning("[Scheduler] 服務已在運行中。")
            return

        logging.info("[Scheduler] 正在啟動...")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="SchedulerThread", daemon=True)
        self._thread.start()

    def stop(self):
        """停止排程器服務並等待背景執行緒終止。"""
        if not self._thread or not self._thread.is_alive():
            logging.warning("[Scheduler] 服務未運行。")
            return

        logging.info("[Scheduler] 正在停止...")
        self._stop_event.set()
        self._thread.join(timeout=self._check_interval_seconds + 5)
        if self._thread.is_alive():
            logging.error("[Scheduler] 執行緒未能優雅地停止。")
        else:
            logging.info("[Scheduler] 已成功停止。")
        self._thread = None
