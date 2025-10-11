# src/moshousapient/streams/video_streamer.py
"""
定義了 VideoStreamer，負責從 RTSP 源讀取和解碼影像流。
它使用 FFmpeg 作為後端，在一個獨立的執行緒中持續生成影像幀，
並將其分發到一個或多個佇列中，供後續的處理器消費。
"""

# 1. 標準庫導入
import logging
import subprocess
import threading
import time
from queue import Queue
from typing import List, Optional

# 2. 第三方庫導入
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
        self.process: Optional[subprocess.Popen] = None
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
        if not self.is_alive():
            logging.critical("[串流器] FFmpeg 即時分析程序啟動失敗或立即退出。請檢查 RTSP URL 和網路連線。")
            # 讓 app_orchestrator 根據 is_alive() 的狀態來決定是否終止應用
            self.stop()

    def _update(self):
        """在背景執行緒中運行的主函式，負責讀取 FFmpeg 輸出並分發幀。"""
        logging.debug("[串流器] 正在啟動 FFmpeg 即時分析程序...")
        bytes_per_frame = self.width * self.height * 3
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=bytes_per_frame,
        )
        logging.info("[串流器] FFmpeg 即時分析程序已成功啟動。")

        while not self.stopped:
            if self.process.stdout:
                raw_frame = self.process.stdout.read(bytes_per_frame)
                if len(raw_frame) == bytes_per_frame:
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
                    item = {"frame": frame, "time": time.time()}
                    for q in self.queues:
                        if not q.full():
                            q.put(item, block=False)
                else:
                    if self.process.poll() is not None:
                        logging.warning("[串流器] FFmpeg 即時分析程序已終止。")
                        break
            else:
                # stdout 可能不存在如果 Popen 失敗
                logging.error("[串流器] FFmpeg 程序 stdout 不可用。")
                break

        if self.process and self.process.poll() is None:
            self.process.kill()

        if self.process and self.process.stderr:
            stderr = self.process.stderr.read().decode("utf-8", errors="ignore")
            if stderr:
                logging.error(f"[串流器] FFmpeg 即時分析程序 stderr:\n{stderr}")
        logging.debug("[串流器] 生產者執行緒已停止。")

    def stop(self):
        """停止讀取執行緒。"""
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def is_alive(self) -> bool:
        """檢查讀取執行緒和 FFmpeg 程序是否仍在運行。"""
        process_alive = self.process is not None and self.process.poll() is None
        thread_alive = self.thread is not None and self.thread.is_alive()
        return thread_alive and process_alive
