# src/moshousapient/services/ingestion_service.py
"""
提供數據攝取服務，用於監控外部數據源並將其轉換為系統內部的任務。

目前，此服務實現了對本地檔案系統目錄的監控。
當有新的影片檔案被添加到受監控的目錄時，它會自動為該檔案創建一個
新的分析任務，並將其提交到任務佇列中。

此服務被設計為一個獨立的、常駐的背景執行緒。
"""

# 1. 標準庫導入
import logging
import pickle
import time
from pathlib import Path
import threading
from typing import Optional

# 2. 第三方庫導入
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

# 3. 本專案相對導入
from ..services.task_queue_service import TaskQueueService


class _VideoFileEventHandler(PatternMatchingEventHandler):
    """
    一個專門處理影片檔案創建事件的處理器。
    """
    # 只關心常見的影片檔案格式
    VIDEO_PATTERNS = ["*.mp4", "*.avi", "*.mov", "*.mkv"]

    def __init__(self, service_instance: 'IngestionService'):
        super().__init__(patterns=self.VIDEO_PATTERNS, ignore_directories=True, case_sensitive=False)
        self._service_instance = service_instance

    def on_created(self, event):
        """
        當一個新檔案被創建時觸發。
        """
        logging.debug(f"[_VideoFileEventHandler] 偵測到新檔案: {event.src_path}")
        # 呼叫主服務類別的方法來處理這個新檔案
        self._service_instance.handle_new_file(Path(event.src_path))


class IngestionService:
    """
    監控指定的目錄，並為新增的影片檔案自動創建處理任務。
    """

    def __init__(self, watch_directory: Path, task_queue: TaskQueueService):
        """
        初始化攝取服務。

        :param watch_directory: 需要被監控的目錄路徑。
        :param task_queue: 用於提交新任務的任務佇列服務實例。
        """
        self._watch_directory = watch_directory
        self._task_queue = task_queue
        self._observer: Optional[Observer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 確保監控目錄存在
        self._watch_directory.mkdir(parents=True, exist_ok=True)

        logging.debug(f"[IngestionService] 已初始化，準備監控目錄: '{self._watch_directory}'")

    def _run_observer(self):
        """
        在背景執行緒中運行的目標函式，負責啟動和管理 Observer。
        """
        event_handler = _VideoFileEventHandler(self)
        self._observer = Observer()
        self._observer.schedule(event_handler, str(self._watch_directory), recursive=False)
        self._observer.start()
        logging.info(f"[IngestionService] 監控服務已在背景啟動。")

        try:
            while not self._stop_event.is_set():
                time.sleep(1)
        finally:
            self._observer.stop()
            self._observer.join()
            logging.debug("[IngestionService] Observer 已停止。")

    def start(self):
        """
        在一個獨立的背景執行緒中啟動監控服務。
        """
        if self._thread is not None and self._thread.is_alive():
            logging.warning("[IngestionService] 服務已在運行中，忽略啟動請求。")
            return

        logging.info(f"[IngestionService] 正在啟動...")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_observer, name="IngestionServiceThread", daemon=True)
        self._thread.start()

    def stop(self):
        """
        停止監控服務並等待背景執行緒終止。
        """
        if self._thread is None or not self._thread.is_alive():
            logging.warning("[IngestionService] 服務未運行，忽略停止請求。")
            return

        logging.info("[IngestionService] 正在停止...")
        self._stop_event.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            logging.error("[IngestionService] 執行緒未能優雅地停止。")
        else:
            logging.info("[IngestionService] 已成功停止。")
        self._thread = None
        self._observer = None

    def handle_new_file(self, file_path: Path):
        """
        處理被偵測到的新檔案，並將其作為任務提交。

        包含一個簡單的延遲和檔案大小檢查，以確保檔案已完全寫入。
        """
        try:
            # 等待一小段時間，以防檔案正在被寫入
            time.sleep(2)

            if not file_path.exists() or file_path.stat().st_size == 0:
                logging.warning(f"[IngestionService] 檔案 '{file_path.name}' 是空的或已被刪除，已忽略。")
                return

            logging.info(f"[IngestionService] 發現有效新檔案 '{file_path.name}'，正在創建任務。")

            task_payload = {
                'task_type': 'file_inference',
                'video_path': str(file_path.resolve())  # 使用絕對路徑
            }
            payload_bytes = pickle.dumps(task_payload)

            task_id = self._task_queue.add_task(payload_bytes)

            if task_id:
                logging.info(f"[IngestionService] 已成功為檔案 '{file_path.name}' 創建任務，任務 ID: {task_id}")
            else:
                logging.error(f"[IngestionService] 為檔案 '{file_path.name}' 創建任務失敗。")

        except Exception as e:
            logging.error(f"[IngestionService] 處理檔案 '{file_path.name}' 時發生錯誤: {e}", exc_info=True)