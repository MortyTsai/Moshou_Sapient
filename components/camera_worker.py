# components/camera_worker.py
import threading
from queue import Queue
import logging
from typing import Dict

from ultralytics import YOLO
from components.video_streamer import VideoStreamer
from components.event_processor import frame_consumer, inference_worker
from components.trackers.sort_tracker import Sort
from config import Config

class CameraWorker:
    """
    封裝單一攝影機所有相關元件和執行緒的類別。
    """
    def __init__(self, camera_config: dict, model: YOLO, class_names: Dict[int, str], notifier=None):
        # 1. 基礎屬性
        self.config = camera_config
        self.model = model
        self.class_names = class_names
        self.notifier = notifier
        self.name = self.config.get("name", "Camera-Default")

        # 2. 狀態管理
        self.stop_event = threading.Event()
        self.shared_state_lock = threading.Lock()
        self.shared_state = {'person_detected': False, 'tracked_objects': []}

        # 3. 資料佇列
        queue_size = int(Config.TARGET_FPS * (Config.PRE_EVENT_SECONDS + Config.POST_EVENT_SECONDS) * 2.0)
        self.consumer_queue = Queue(maxsize=queue_size)
        self.inference_queue = Queue(maxsize=2)

        # 4. 核心元件
        self.video_streamer = VideoStreamer(src=self.config['rtsp_url'],
                                            width=Config.ENCODE_WIDTH, height=Config.ENCODE_HEIGHT,
                                            use_udp=(self.config.get("transport_protocol", "udp").lower() == 'udp'))
        self.tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.3)

        # 5. 執行緒
        self.consumer_thread = threading.Thread(target=frame_consumer, name=f"{self.name}-Consumer",
                                                args=(self.consumer_queue, self.shared_state, self.stop_event, self.notifier, self.shared_state_lock))
        self.inference_thread = threading.Thread(target=inference_worker, name=f"{self.name}-Inference",
                                                 args=(self.inference_queue, self.shared_state, self.stop_event, self.shared_state_lock, self.model, self.class_names, self.tracker))
        self.threads = [self.consumer_thread, self.inference_thread]

    def start(self):
        logging.info(f"[{self.name}] 正在啟動...")
        for t in self.threads:
            t.start()
        self.video_streamer.start(self.consumer_queue, self.inference_queue)
        logging.info(f"[{self.name}] 所有執行緒已啟動, 系統現已全面運作。")

    def stop(self):
        logging.info(f"\n[{self.name}] 正在關閉...")
        self.stop_event.set()
        if self.video_streamer:
            self.video_streamer.stop()
        for t in self.threads:
            if t.is_alive():
                logging.info(f"[{self.name}] 等待 {t.name} 執行緒結束...")
                t.join(timeout=Config.THREAD_JOIN_TIMEOUT)
                if t.is_alive():
                    logging.warning(f"[{self.name}] {t.name} 執行緒關閉超時。")
        logging.info(f"[{self.name}] 已安全關閉。")

    def is_alive(self):
        streamer_thread_alive = self.video_streamer.thread and self.video_streamer.thread.is_alive()
        return all(t.is_alive() for t in self.threads) and streamer_thread_alive