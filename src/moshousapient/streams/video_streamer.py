# src/moshousapient/streams/video_streamer.py
"""
定義了 VideoStreamer，負責從 RTSP 源讀取和解碼影像流。
它使用 FFmpeg 作為後端，在一個獨立的執行緒中持續生成影像幀，
並將其分發到一個或多個佇列中，供後續的處理器消費。
"""

import logging
import subprocess
import threading
import time
from queue import Queue
from typing import List, Optional

import numpy as np


class VideoStreamer:
    """
    一個使用 FFmpeg 和獨立執行緒的 RTSP 影像流讀取器。
    """

    def __init__(self, camera_config: dict, width: int, height: int):
        """
        初始化 VideoStreamer。
        """
        self.camera_config = camera_config
        self.width = width
        self.height = height
        self.stopped = False
        self.thread: Optional[threading.Thread] = None
        self.queues: List[Queue] = []
        self.command = self._build_ffmpeg_command()
        logging.debug("VideoStreamer 已初始化。")

    def _build_ffmpeg_command(self) -> list:
        """根據設定構建 FFmpeg 命令列。"""
        source_rtsp = self.camera_config.get("rtsp_url")
        use_udp = self.camera_config.get("transport_protocol", "udp").lower() == "udp"
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        command.extend(["-hwaccel", "cuda", "-c:v", "h264_cuvid"])
        if use_udp:
            command.extend(["-rtsp_transport", "udp", "-rtbufsize", "50M"])
        else:
            command.extend(["-rtsp_transport", "tcp", "-rtbufsize", "20M"])
        command.extend(["-i", source_rtsp])
        command.extend(
            [
                "-vf",
                f"scale={self.width}:{self.height}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ]
        )
        return command

    def start(self, *queues: Queue):
        """啟動影像流讀取執行緒。"""
        self.queues = list(queues)
        self.thread = threading.Thread(target=self._update, name="VideoStreamThread", daemon=True)
        self.thread.start()
        logging.debug("[串流器] 生產者執行緒已啟動。")
        time.sleep(5)  # 給予 FFmpeg 啟動和緩衝時間
        if not self.thread.is_alive():
            raise ConnectionError("FFmpeg 即時分析程序啟動失敗。")

    def _update(self):
        """在背景執行緒中運行的主函式，負責讀取 FFmpeg 輸出並分發幀。"""
        logging.debug("[串流器] 正在啟動 FFmpeg 即時分析程序...")
        bytes_per_frame = self.width * self.height * 3
        process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=bytes_per_frame,
        )
        logging.info("[串流器] FFmpeg 即時分析程序已成功啟動。")

        while not self.stopped:
            raw_frame = process.stdout.read(bytes_per_frame)
            if len(raw_frame) == bytes_per_frame:
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
                item = {"frame": frame, "time": time.time()}
                for q in self.queues:
                    if not q.full():
                        q.put(item, block=False)
            else:
                if process.poll() is not None:
                    logging.warning("[串流器] FFmpeg 即時分析程序已終止。")
                    break

        if process.poll() is None:
            process.kill()

        stderr = process.stderr.read().decode("utf-8", errors="ignore")
        if stderr:
            logging.error(f"[串流器] FFmpeg 即時分析程序 stderr:\n{stderr}")
        logging.debug("[串流器] 生產者執行緒已停止。")

    def stop(self):
        """停止讀取執行緒。"""
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def is_alive(self) -> bool:
        """檢查讀取執行緒是否仍在運行。"""
        return self.thread and self.thread.is_alive()
