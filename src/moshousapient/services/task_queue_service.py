# src/moshousapient/services/task_queue_service.py
"""
此模組提供 TaskQueueService，用於管理基於 SQLite 的持久化任務佇列。
它負責任務的加入、預留、完成和失敗處理，確保任務在各個服務間的可靠傳遞。
"""

# 1. 標準庫導入
import sqlite3
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# 3. 本專案相對導入
from ..settings import settings


class TaskQueueService:
    """
    封裝所有與 SQLite 任務佇列互動的服務類別。

    此類別被設計為執行緒安全的，允許多個 Worker 同時存取。
    """

    def __init__(self, db_path: Path = settings.DATA_DIR / "tasks.db"):
        """
        初始化任務佇列服務。

        :param db_path: SQLite 資料庫檔案的路徑。
        """
        self.db_path = db_path
        self.local = threading.local()
        self._initialize_database()
        logging.info(f"[TaskQueue] 任務佇列服務已初始化，資料庫位於: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """為當前執行緒獲取或建立一個資料庫連線。"""
        if not hasattr(self.local, 'conn') or self.local.conn is None:
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
        if hasattr(self.local, 'conn') and self.local.conn is not None:
            self.local.conn.close()
            self.local.conn = None
            logging.debug("[TaskQueue] 當前執行緒的資料庫連線已關閉。")

    def _initialize_database(self):
        """初始化資料庫，如果任務資料表不存在，則建立它。"""
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
                    reserved_at TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks (status);")
            conn.commit()
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 初始化資料庫時發生錯誤: {e}", exc_info=True)
            self.close_connection()
            raise

    def add_task(self, payload: bytes) -> Optional[int]:
        """
        將一個新任務加入佇列。

        :param payload: 已經被序列化的二進位數據。
        :return: 成功時返回新任務的 ID，失敗時返回 None。
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO tasks (payload, status) VALUES (?, 'pending')",
                (payload,)
            )
            conn.commit()
            task_id = cursor.lastrowid
            logging.debug(f"[TaskQueue] 已成功新增任務 ID: {task_id}")
            return task_id
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 新增任務時發生錯誤: {e}", exc_info=True)
            self.close_connection()
            return None

    def reserve_task(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        以原子方式預留一個待處理的任務。

        :param worker_id: 正在預留此任務的 Worker 的唯一識別碼。
        :return: 一個包含任務資訊的字典 (id, payload)，如果沒有待處理任務則返回 None。
        """
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, payload FROM tasks WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
                )
                task_row = cursor.fetchone()

                if task_row is None:
                    return None

                task_id = task_row['id']
                payload = task_row['payload']

                cursor.execute(
                    """
                    UPDATE tasks
                    SET status = 'processing', worker_id = ?, reserved_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (worker_id, datetime.now(timezone.utc), task_id)
                )

                if cursor.rowcount > 0:
                    task_data = {'id': task_id, 'payload': payload}
                    logging.debug(f"[TaskQueue] Worker '{worker_id}' 已預留任務 ID: {task_id}")
                    return task_data
                else:
                    return None

        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 預留任務時發生錯誤: {e}", exc_info=True)
            return None

    def complete_task(self, task_id: int):
        """
        將一個已完成的任務從佇列中移除。

        :param task_id: 要移除的任務 ID。
        """
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            logging.debug(f"[TaskQueue] 已完成並移除任務 ID: {task_id}")
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 完成任務 ID {task_id} 時發生錯誤: {e}", exc_info=True)

    def fail_task(self, task_id: int):
        """
        標記一個任務失敗，將其狀態重設為 'pending' 以供重試。

        :param task_id: 失敗的任務 ID。
        """
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE tasks SET status = 'pending', worker_id = NULL, reserved_at = NULL WHERE id = ?",
                    (task_id,)
                )
            logging.warning(f"[TaskQueue] 任務 ID {task_id} 處理失敗，已重新排入佇列。")
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 標記任務 ID {task_id} 失敗時發生錯誤: {e}", exc_info=True)

    def requeue_stale_tasks(self, timeout_seconds: int = 300):
        """
        查找並重新排入因 Worker 崩潰而卡在 'processing' 狀態的過期任務。

        :param timeout_seconds: 任務被視為過期的秒數。預設為 5 分鐘。
        """
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()

                timeout_point = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)

                cursor.execute(
                    "SELECT id, worker_id FROM tasks WHERE status = 'processing' AND reserved_at < ?",
                    (timeout_point,)
                )
                stale_tasks = cursor.fetchall()

                if not stale_tasks:
                    logging.debug("[TaskQueue] 未發現過期任務。")
                    return

                task_ids_to_requeue = [task['id'] for task in stale_tasks]
                logging.warning(f"[TaskQueue] 發現 {len(stale_tasks)} 個過期任務，將它們重新排入佇列。")
                for task in stale_tasks:
                    logging.warning(f"  - 任務 ID: {task['id']}, 原 Worker: {task['worker_id']}")

                cursor.execute(
                    f"""
                    UPDATE tasks
                    SET status = 'pending', worker_id = NULL, reserved_at = NULL
                    WHERE id IN ({','.join('?' for _ in task_ids_to_requeue)})
                    """,
                    task_ids_to_requeue
                )
        except sqlite3.Error as e:
            logging.error(f"[TaskQueue] 重新排入過期任務時發生錯誤: {e}", exc_info=True)