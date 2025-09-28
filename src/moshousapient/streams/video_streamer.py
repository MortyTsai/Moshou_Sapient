# src/moshousapient/streams/video_streamer.py (Definitive Final Version)
import logging
import subprocess
import threading
import time
from queue import Queue
from typing import List
import numpy as np
import cv2


class VideoStreamer:
    def __init__(self, camera_config: dict, width: int, height: int):
        self.camera_config = camera_config
        self.width = width
        self.height = height
        self.stopped = False
        self.thread = None
        self.queues: List[Queue] = []

        # 獲取攝影機的原始解析度 (硬編碼，待改進)
        # TODO: 使用 ffprobe 動態獲取
        self.source_width, self.source_height = 2304, 1296
        self.bytes_per_frame = int(self.source_width * self.source_height * 1.5)  # YUV420p

        self.command = self._build_ffmpeg_command()

    def _build_ffmpeg_command(self) -> list:
        source_rtsp = self.camera_config.get('rtsp_url')
        use_udp = self.camera_config.get('transport_protocol', 'udp').lower() == 'udp'

        command = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
        command.extend(['-hwaccel', 'cuda', '-c:v', 'h264_cuvid'])

        if use_udp:
            command.extend(['-rtsp_transport', 'udp', '-rtbufsize', '50M'])
        else:
            command.extend(['-rtsp_transport', 'tcp', '-rtbufsize', '20M'])

        command.extend(['-i', source_rtsp])
        command.extend(['-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'])
        return command

    def start(self, *queues: Queue):
        self.queues = list(queues)
        self.thread = threading.Thread(target=self.update, name="VideoStreamThread", daemon=True)
        self.thread.start()
        logging.info("[串流器] 生產者執行緒已啟動。")
        time.sleep(5)
        if not self.thread.is_alive():
            raise ConnectionError("FFmpeg 即時分析程序啟動失敗。")

    def update(self):
        logging.info("[串流器] 正在啟動 FFmpeg 即時分析程序...")
        process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   bufsize=self.bytes_per_frame)
        logging.info("[串流器] FFmpeg 即時分析程序已成功啟動。")

        while not self.stopped:
            raw_frame = process.stdout.read(self.bytes_per_frame)
            if len(raw_frame) == self.bytes_per_frame:
                yuv_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (int(self.source_height * 1.5), self.source_width))
                bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                resized_frame = cv2.resize(bgr_frame, (self.width, self.height))

                item = {'frame': resized_frame, 'time': time.time()}
                for q in self.queues:
                    if not q.full(): q.put(item, block=False)
            else:
                if process.poll() is not None:
                    logging.warning("[串流器] FFmpeg 即時分析程序已終止。")
                    break

        if process.poll() is None: process.kill()
        stderr = process.stderr.read().decode('utf-8', errors='ignore')
        if stderr: logging.error(f"[串流器] FFmpeg 即時分析程序 stderr:\n{stderr}")
        logging.info("[串流器] 生產者執行緒已停止。")

    def stop(self):
        self.stopped = True
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=5)

    def is_alive(self) -> bool:
        return self.thread and self.thread.is_alive()