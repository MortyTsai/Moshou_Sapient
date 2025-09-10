# video_streamer.py
import subprocess
import threading
import numpy as np
import time
import logging
from ..config import Config

class VideoStreamer:
    """
    使用 FFmpeg 從 RTSP 來源或本地檔案讀取影像串流，並將幀放入佇列中。
    """

    def __init__(self, src, width, height, use_udp=True):
        self.src = src
        self.width = width
        self.height = height
        self.stopped = False
        self.thread = None
        self.consumer_queue = None
        self.inference_queue = None
        self.command = ['ffmpeg', '-hide_banner', '-loglevel', 'error']

        if Config.VIDEO_SOURCE_TYPE == "FILE":
            logging.info(f"影像讀取器: 初始化檔案串流 (來源: {self.src})。使用 -re 參數模擬即時幀率。")
            self.command.extend(['-re', '-i', self.src])

        elif Config.VIDEO_SOURCE_TYPE == "RTSP":
            protocol = "UDP" if use_udp else "TCP"
            logging.info(f"影像讀取器: 初始化 RTSP 串流 (協定: {protocol})，解析度: {width}x{height}")
            if use_udp:
                self.command.extend(
                    ['-rtsp_transport', 'udp', '-probesize', '5M', '-analyzeduration', '5M', '-i', self.src])
            else:
                self.command.extend(['-rtsp_transport', 'tcp', '-rtbufsize', '20M', '-i', self.src])

        self.command.extend(['-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'])

    def start(self, consumer_queue, inference_queue):
        self.consumer_queue = consumer_queue
        self.inference_queue = inference_queue
        self.thread = threading.Thread(target=self.update, name="VideoStreamThread", args=())
        self.thread.daemon = True
        self.thread.start()
        logging.info("影像讀取器: 生產者執行緒已啟動。")
        time.sleep(3)
        if not self.thread.is_alive():
            if Config.VIDEO_SOURCE_TYPE == "RTSP":
                raise ConnectionError("FFmpeg 程序啟動失敗，請檢查 RTSP URL 與攝影機連線。")
            else:
                raise ConnectionError(f"FFmpeg 程序啟動失敗，請檢查影片檔案路徑是否正確: {self.src}")
        return self

    def update(self):
        bytes_per_frame = self.width * self.height * 3
        process = None
        try:
            logging.info(f"影像讀取器: 正在啟動 FFmpeg 程序...")
            process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       bufsize=bytes_per_frame)
            logging.info("影像讀取器: FFmpeg 程序已成功啟動。")

            while not self.stopped:
                raw_frame = process.stdout.read(bytes_per_frame)
                if len(raw_frame) == bytes_per_frame:
                    frame = np.frombuffer(raw_frame, np.uint8).reshape((self.height, self.width, 3))
                    item = {'frame': frame, 'time': time.time()}

                    if self.consumer_queue and not self.consumer_queue.full():
                        self.consumer_queue.put(item, block=False)

                    if self.inference_queue and not self.inference_queue.full():
                        self.inference_queue.put(item, block=False)

                if process.poll() is not None:
                    logging.warning("影像讀取器: FFmpeg 程序已終止。")
                    if Config.VIDEO_SOURCE_TYPE == "FILE":
                        logging.info("影像讀取器: 影片檔案已讀取完畢。")
                    break

        except Exception as e:
            logging.error(f"影像讀取器: 主更新迴圈發生錯誤: {e}", exc_info=True)
        finally:
            if process and process.poll() is None:
                process.kill()
                process.wait()
                logging.info("影像讀取器: 已終止 FFmpeg 程序。")

            if process and process.stderr:
                stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                if stderr_output:
                    logging.error(f"影像讀取器 FFmpeg stderr:\n{stderr_output.strip()}")

            logging.info("影像讀取器: 生產者執行緒正在停止。")

    def stop(self):
        logging.info("影像讀取器: 正在停止生產者執行緒...")
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=Config.THREAD_JOIN_TIMEOUT)
            if self.thread.is_alive():
                logging.warning("影像讀取器: 生產者執行緒關閉超時。")
        logging.info("影像讀取器: 生產者執行緒已停止。")