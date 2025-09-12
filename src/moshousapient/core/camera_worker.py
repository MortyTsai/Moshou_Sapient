# src/moshousapient/core/camera_worker.py
import logging, yaml, threading
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
        self.active_recorders = []
        self.shared_state = {'person_detected': False, 'tracked_objects': []}
        self.shared_state_lock = threading.Lock()
        self.inference_queue = Queue(maxsize=2)
        buffer_size = int(Config.TARGET_FPS * (Config.PRE_EVENT_SECONDS + Config.POST_EVENT_SECONDS) * 2.0)
        self.event_queue = Queue(maxsize=buffer_size)
        self.video_streamer = VideoStreamer(
            src=self.config['rtsp_url'],
            width=Config.ENCODE_WIDTH,
            height=Config.ENCODE_HEIGHT,
            use_udp=(self.config.get("transport_protocol", "udp").lower() == 'udp')
        )
        tracker = self._initialize_tracker()
        self.inference_processor = InferenceProcessor(
            frame_queue=self.inference_queue,
            shared_state=self.shared_state,
            state_lock=self.shared_state_lock,
            model=model,
            reid_model=reid_model,
            tracker=tracker,
            name=f"{self.name}-Inference"
        )
        self.event_processor = EventProcessor(
            frame_queue=self.event_queue,
            shared_state=self.shared_state,
            state_lock=self.shared_state_lock,
            notifier=self.notifier,
            active_recorders=self.active_recorders,
            name=f"{self.name}-Event"
        )
        self.processors = [self.inference_processor, self.event_processor]

    def _initialize_tracker(self):
        try:
            with open(Config.TRACKER_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f)
            tracker_args = SimpleNamespace(**cfg_dict)
            from ultralytics.trackers import BOTSORT
            logging.info(f"[{self.name}] 已成功解析追蹤器設定檔: {Config.TRACKER_CONFIG_PATH}")
            return BOTSORT(args=tracker_args, frame_rate=30)
        except Exception as e:
            logging.error(f"[{self.name}] 解析追蹤器設定檔或建立追蹤器時發生錯誤: {e}", exc_info=True)
            return None

    def start(self):
        logging.info(f"[{self.name}] 正在啟動...")
        for processor in self.processors:
            processor.start()
        self.video_streamer.start(self.event_queue, self.inference_queue)

    def stop(self):
        logging.info(f"[{self.name}] 正在關閉...")
        if self.video_streamer:
            self.video_streamer.stop()
        for processor in self.processors:
            processor.stop()
        if self.active_recorders:
            running_recorders = [r for r in self.active_recorders if r.is_alive()]
            if running_recorders:
                logging.info(f"[{self.name}] {len(running_recorders)} 個事件錄影執行緒正在背景處理中...")
        logging.info(f"[{self.name}] 已安全關閉。")

    def is_alive(self) -> bool:
        return self.video_streamer and self.video_streamer.is_alive()