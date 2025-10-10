# src/moshousapient/services/task_queue_service.py
"""
此模組提供 TaskQueueService，用於管理基於 SQLite 的持久化任務佇列。

它負責任務的加入、預留、完成和失敗處理，確保任務在各個服務間的可靠傳遞。
此類別被設計為執行緒安全的，允許多個 Worker 同時存取。
"""

# 1. 標準庫導入
import logging
import pickle
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# 2. 第三方庫導入
# (無)

# 3. 本專案相對導入
from ..configs.settings_config import settings


class TaskQueueService:
    """
    封裝所有與 SQLite 任務佇列互動的服務類別。
    """

    def __init__(self, db_path: Path = settings.DATA_DIR / "tasks.db"):
        """
        初始化任務佇列服務。

        :param db_path: SQLite 資料庫檔案的路徑。
        """
        self.db_path = db_path
        self.local = threading.local()
        self.max_retries = 3
        self._initialize_database()
        logging.debug(f"[TaskQueue] 任務佇列服務已初始化，資料庫位於: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """為當前執行緒獲取或建立一個資料庫連線。"""
        if not hasattr(self.local, "conn") or self.local.conn is None:
            try:
                self.local.conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
                self.local.conn.execute("PRAGMA journal_mode=WAL;")
                self.local.conn.row_factory = sqlite3.Row
            except sqlite3.Error as e:
                logging.critical(f"[TaskQueue] 無法連線至任務佇列資料庫: {e}", exc_info=True)
                raise
        return self.local.conn

    def close_connection(self):
        """關閉當前執行緒的資料庫連線。"""
        if hasattr(self.local, "conn") and self.local.conn is not None:
            self.local.conn.close()
            self.local.conn = None

    def _initialize_database(self):
        """初始化資料庫，如果任務資料表不存在，則建立它，並處理結構遷移。"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload BLOB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    worker_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reserved_at TIMESTAMP,
                    retry_count INTEGER DEFAULT 0,
                    last_error TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks (status);")

            cursor.execute("PRAGMA table_info(tasks)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "retry_count" not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN retry_count INTEGER DEFAULT 0")
            if "last_error" not in columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN last_error TEXT")

            conn.commit()
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 初始化或升級資料庫時發生錯誤: {e}", exc_info=True)
            self.close_connection()
            raise

    def add_task(self, payload: bytes) -> Optional[int]:
        """將一個新任務加入佇列。"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO tasks (payload, status) VALUES (?, 'pending')", (payload,))
            conn.commit()
            task_id = cursor.lastrowid
            logging.debug(f"[TaskQueue] 已成功新增任務 ID: {task_id}")
            return task_id
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 新增任務時發生錯誤: {e}", exc_info=True)
            self.close_connection()
            return None

    def has_pending_task_by_type(self, task_type: str) -> bool:
        """
        檢查是否存在指定類型的待處理任務。

        :param task_type: 要查詢的任務類型 (例如 'file_inference')。
        :return: 如果存在至少一個匹配的待處理任務，則返回 True。
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT payload FROM tasks WHERE status = 'pending'")

            for row in cursor.fetchall():
                try:
                    payload_dict = pickle.loads(row["payload"])
                    if payload_dict.get("task_type") == task_type:
                        return True
                except (pickle.UnpicklingError, KeyError):
                    # 忽略無法解析或格式不符的 payload
                    continue
            return False
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 查詢待處理任務時發生錯誤: {e}", exc_info=True)
            return False

    def reserve_task(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """以原子方式預留一個待處理的任務。"""
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, payload FROM tasks WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1")
                task_row = cursor.fetchone()
                if task_row is None:
                    return None

                task_id = task_row["id"]
                payload = task_row["payload"]
                cursor.execute(
                    "UPDATE tasks SET status = 'processing', worker_id = ?, reserved_at = ? WHERE id = ? AND status = 'pending'",
                    (worker_id, datetime.now(timezone.utc), task_id),
                )
                if cursor.rowcount > 0:
                    task_data = {"id": task_id, "payload": payload}
                    logging.debug(f"[TaskQueue] Worker '{worker_id}' 已預留任務 ID: {task_id}")
                    return task_data
                else:
                    # 任務可能剛好被另一個 worker 預留，這是一個正常的競爭條件
                    return None
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 預留任務時發生錯誤: {e}", exc_info=True)
            return None

    def complete_task(self, task_id: int):
        """將一個已完成的任務從佇列中移除。"""
        try:
            conn = self._get_connection()
            with conn:
                conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            logging.debug(f"[TaskQueue] 已完成並移除任務 ID: {task_id}")
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 完成任務 ID {task_id} 時發生錯誤: {e}", exc_info=True)

    def fail_task(self, task_id: int, error_message: str = "Unknown error", requeue: bool = False):
        """標記一個任務失敗，或根據標誌將其直接重新排隊。"""
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()

                if requeue:
                    # 直接將任務狀態改回 pending，不增加重試計數，用於任務釋放
                    cursor.execute(
                        "UPDATE tasks SET status = 'pending', worker_id = NULL, reserved_at = NULL, last_error = ? WHERE id = ?",
                        (error_message, task_id),
                    )
                    return

                cursor.execute("SELECT retry_count FROM tasks WHERE id = ?", (task_id,))
                result = cursor.fetchone()
                if result is None:
                    logging.warning(f"[TaskQueue] 嘗試標記失敗時找不到任務 ID {task_id}。可能已被處理。")
                    return

                current_retries = result["retry_count"]
                if current_retries + 1 >= self.max_retries:
                    final_error_msg = f"Max retries reached. Last error: {error_message}"
                    logging.error(
                        f"[TaskQueue] 任務 ID {task_id} 已達到最大重試次數 ({self.max_retries})。將標記為永久失敗。"
                    )
                    cursor.execute(
                        "UPDATE tasks SET status = 'failed', last_error = ? WHERE id = ?",
                        (final_error_msg, task_id),
                    )
                else:
                    logging.warning(
                        f"[TaskQueue] 任務 ID {task_id} 處理失敗 (嘗試 {current_retries + 1}/{self.max_retries})，將重新排隊。"
                    )
                    cursor.execute(
                        "UPDATE tasks SET status = 'pending', worker_id = NULL, reserved_at = NULL, retry_count = retry_count + 1, last_error = ? WHERE id = ?",
                        (error_message, task_id),
                    )
        except sqlite3.Error as e:
            logging.error(
                f"[TaskQueue] 標記任務 ID {task_id} 失敗時發生資料庫錯誤: {e}",
                exc_info=True,
            )

    def delete_stale_tasks(self, timeout_seconds: int = 600) -> int:
        """查找並刪除因 Worker 崩潰而卡在 'processing' 狀態的過期任務。"""
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                timeout_point = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
                cursor.execute(
                    "SELECT id, worker_id FROM tasks WHERE status = 'processing' AND reserved_at < ?",
                    (timeout_point,),
                )
                stale_tasks = cursor.fetchall()
                if not stale_tasks:
                    logging.debug("[TaskQueue] 系統啟動檢查：未發現需要清理的過期任務。")
                    return 0

                task_ids_to_delete = [task["id"] for task in stale_tasks]
                logging.warning(f"[TaskQueue] 發現 {len(stale_tasks)} 個因上次異常關閉而遺留的任務，將其清理。")
                placeholders = ",".join("?" for _ in task_ids_to_delete)
                cursor.execute(
                    f"DELETE FROM tasks WHERE id IN ({placeholders})",
                    task_ids_to_delete,
                )
                return len(task_ids_to_delete)
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 清理過期任務時發生錯誤: {e}", exc_info=True)
            return 0

    def get_pending_or_processing_count(self) -> int:
        """獲取狀態為 'pending' 或 'processing' 的任務總數。"""
        count = 0
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(id) FROM tasks WHERE status = 'pending' OR status = 'processing'")
            result = cursor.fetchone()
            if result:
                count = result[0]
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 獲取任務計數時發生錯誤: {e}", exc_info=True)
        finally:
            self.close_connection()
        return count

    def reset_stale_processing_tasks(self, timeout_seconds: int) -> int:
        """
        查找並重置因 Worker 異常終止而卡在 'processing' 狀態的過期任務。
        與 delete_stale_tasks 不同，此方法將任務狀態重置回 'pending' 而不是刪除。

        :param timeout_seconds: 任務被預留超過此秒數後被視為過期。
        :return: 被重置的任務數量。
        """
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                timeout_point = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)

                # 查找所有超時的 'processing' 任務
                cursor.execute(
                    "SELECT id FROM tasks WHERE status = 'processing' AND reserved_at < ?",
                    (timeout_point,),
                )
                stale_tasks = cursor.fetchall()
                if not stale_tasks:
                    return 0

                task_ids_to_reset = [task["id"] for task in stale_tasks]
                logging.warning(
                    f"[TaskQueue] 發現 {len(task_ids_to_reset)} 個過期的 'processing' 任務，將其狀態重置為 'pending'。"
                )

                placeholders = ",".join("?" for _ in task_ids_to_reset)
                cursor.execute(
                    f"UPDATE tasks SET status = 'pending', worker_id = NULL, reserved_at = NULL, last_error = 'Reset by Scheduler' WHERE id IN ({placeholders})",
                    task_ids_to_reset,
                )
                return len(task_ids_to_reset)
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 重置過期任務時發生錯誤: {e}", exc_info=True)
            return 0
