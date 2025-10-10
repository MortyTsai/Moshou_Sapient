# src/moshousapient/utils/video_io_utils.py
"""
提供與影片 I/O (輸入/輸出) 相關的輔助工具和函式。
"""

# 1. 標準庫導入
import json
import logging
import subprocess
import threading
import time
from queue import Queue, Empty
from typing import Tuple, Optional

# 2. 第三方庫導入
import cv2


class ThreadedVideoCapture:
    """
    一個使用獨立執行緒預先讀取影片幀的輔助類別。

    透過將 I/O 操作與主處理邏輯分離，可以顯著減少等待時間，
    特別是在 CPU 密集型的影像處理任務中。
    """

    def __init__(self, source: str, max_queue_size: int = 256):
        """
        初始化 ThreadedVideoCapture。

        :param source: 影片檔案的路徑或串流 URL。
        :param max_queue_size: 內部幀佇列的最大尺寸。
        """
        self.cap = cv2.VideoCapture(source)
        self.q: Queue = Queue(maxsize=max_queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)

    def _update(self):
        """在背景執行緒中持續讀取幀並放入佇列。"""
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    self.q.put((False, None))  # 放入結束信號
                    return
                self.q.put((ret, frame))
            else:
                time.sleep(0.01)  # 佇列已滿，稍作等待

    def start(self) -> "ThreadedVideoCapture":
        """啟動背景讀取執行緒。"""
        self.thread.start()
        return self

    def read(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """從佇列中獲取一幀，此操作為阻塞式。"""
        return self.q.get()

    def is_opened(self) -> bool:
        """檢查影片擷取是否已成功開啟。"""
        return self.cap.isOpened()

    def release(self):
        """停止執行緒並釋放影片擷取資源。"""
        self.stopped = True
        if self.thread.is_alive():
            # 清空佇列以解除 read() 的阻塞
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except Empty:
                    break
            self.thread.join()
        self.cap.release()


def get_video_resolution(video_path: str) -> Optional[Tuple[int, int]]:
    """
    使用 ffprobe 高效地獲取影片的寬度和高度，無需解碼整個檔案。

    :param video_path: 影片檔案的路徑。
    :return: 一個包含 (寬度, 高度) 的元組，如果失敗則返回 None。
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            encoding="utf-8",
        )
        data = json.loads(result.stdout)
        if "streams" in data and len(data["streams"]) > 0:
            width = data["streams"][0].get("width")
            height = data["streams"][0].get("height")
            if width and height:
                return int(width), int(height)
        return None
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as e:
        logging.error(f"[系統] 獲取影片解析度時出錯: {e}")
        return None
