# src/moshousapient/core/camera_worker.py (Definitive Final Version)
import logging
import threading
import yaml
from queue import Queue
from types import SimpleNamespace
from ultralytics import YOLO
from ..streams.video_streamer import VideoStreamer
from ..processors.inference_processor import InferenceProcessor
from ..processors.event_processor import EventProcessor
from ..config import Config


class CameraWorker:
    def __init__(self, camera_config: dict, model: YOLO, reid_model: YOLO, notifier=None):
        self.config = camera_config
        self.name = self.config.get("name", "Camera-Default")
        self.notifier = notifier

        self.shared_state = {'person_detected': False, 'tracked_objects': []}
        self.shared_state_lock = threading.Lock()

        self.video_streamer = VideoStreamer(self.config, Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT)

        self.inference_queue = Queue(maxsize=2)
        buffer_size = int((Config.PRE_EVENT_SECONDS + 1.0) * Config.TARGET_FPS)
        self.event_queue = Queue(maxsize=buffer_size)

        self.processors = [
            InferenceProcessor(self.inference_queue, self.shared_state, self.shared_state_lock, model, reid_model,
                               self._initialize_tracker, f"{self.name}-Inference"),
            EventProcessor(
                frame_queue=self.event_queue,
                shared_state=self.shared_state,
                state_lock=self.shared_state_lock,
                notifier=self.notifier,
                target_fps=Config.TARGET_FPS,
                name=f"{self.name}-Event"
            )
        ]

    def start(self):
        logging.info(f"[{self.name}] 正在啟動...")
        try:
            self.video_streamer.start(self.event_queue, self.inference_queue)
            for processor in self.processors:
                processor.start()
            logging.info(f"[{self.name}] 所有處理執行緒已成功啟動。")
        except Exception as e:
            logging.error(f"[{self.name}] 啟動時發生錯誤: {e}", exc_info=True)
            self.stop()

    def stop(self):
        logging.info(f"[{self.name}] 正在關閉...")
        for processor in self.processors:
            processor.stop()
        if self.video_streamer:
            self.video_streamer.stop()
        logging.info(f"[{self.name}] 已安全關閉。")

    def is_alive(self) -> bool:
        return self.video_streamer and self.video_streamer.is_alive()

    def _initialize_tracker(self):
        try:
            with open(Config.TRACKER_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f)
            tracker_args = SimpleNamespace(**cfg_dict)
            from ultralytics.trackers import BOTSORT
            return BOTSORT(args=tracker_args)
        except Exception as e:
            logging.error(f"[{self.name}] 解析追蹤器設定檔時發生錯誤: {e}", exc_info=True)
            return None