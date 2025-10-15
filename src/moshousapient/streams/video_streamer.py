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
import cv2
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
        self.output_queues: List[Queue] = []
        self.command = self._build_ffmpeg_command()
        self.process: Optional[subprocess.Popen] = None
        self.source_fps: float = self._get_source_fps()
        logging.debug(f"VideoStreamer 已初始化。偵測到來源幀率: {self.source_fps:.2f} FPS。")

    def _get_source_fps(self) -> float:
        """使用 OpenCV 嘗試獲取 RTSP 串流的原始幀率。"""
        cap = None
        fps = 30.0
        try:
            rtsp_url = self.camera_config.get("rtsp_url")
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                logging.warning("[串流器] 無法開啟 RTSP 串流以獲取幀率，將使用預設值 30.0 FPS。")
            else:
                retrieved_fps = cap.get(cv2.CAP_PROP_FPS)
                if retrieved_fps > 0:
                    fps = retrieved_fps
        except Exception:
            logging.exception("[串流器] 獲取來源幀率時發生錯誤，將使用預設值 30.0 FPS。")
        finally:
            if cap:
                cap.release()
        return fps

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

    def add_output_queue(self, queue: Queue):
        """註冊一個新的輸出佇列。"""
        self.output_queues.append(queue)

    def start(self):
        """啟動影像流讀取執行緒。"""
        self.thread = threading.Thread(target=self._update, name="VideoStreamThread", daemon=True)
        self.thread.start()
        logging.debug("[串流器] 生產者執行緒已啟動。")
        time.sleep(5)
        if not self.is_alive():
            logging.critical("[串流器] FFmpeg 即時分析程序啟動失敗或立即退出。請檢查 RTSP URL 和網路連線。")
            self.stop()

    def _run_ffmpeg_process(self):
        """啟動並返回 FFmpeg 子程序。"""
        logging.debug("[串流器] 正在啟動 FFmpeg 即時分析程序...")
        bytes_per_frame = self.width * self.height * 3
        process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=bytes_per_frame,
        )
        logging.info("[串流器] FFmpeg 即時分析程序已成功啟動。")
        return process

    def _broadcast_frames(self):
        """從 FFmpeg 讀取幀並將其廣播到所有輸出佇列。"""
        bytes_per_frame = self.width * self.height * 3
        while not self.stopped:
            if self.process and self.process.stdout:
                raw_frame = self.process.stdout.read(bytes_per_frame)
                if len(raw_frame) == bytes_per_frame:
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
                    item = {"frame": frame, "time": time.time()}
                    for q in self.output_queues:
                        if not q.full():
                            q.put(item)
                else:
                    if self.process.poll() is not None:
                        logging.warning("[串流器] FFmpeg 即時分析程序已終止。")
                        break
            else:
                logging.error("[串流器] FFmpeg 程序 stdout 不可用。")
                break

    def _update(self):
        """在背景執行緒中運行的主函式，負責讀取 FFmpeg 輸出並廣播幀。"""
        self.process = self._run_ffmpeg_process()
        self._broadcast_frames()

        for q in self.output_queues:
            q.put(None)

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
