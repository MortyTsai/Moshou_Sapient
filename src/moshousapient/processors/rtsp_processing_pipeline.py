# src/moshousapient/processors/rtsp_processing_pipeline.py
"""
定義了 RTSPPipeline 類別，用於代表一個完整的、自給自足的 RTSP 影像處理管線。
"""

# 1. 標準庫導入
import logging
import threading
from queue import Queue
from types import SimpleNamespace
from typing import Any

import yaml

# 2. 第三方庫導入
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT

# 3. 本專案導入
from moshousapient.configs.behavior_config import Config
from moshousapient.configs.settings_config import settings
from moshousapient.processors.inference_processor import InferenceProcessor
from moshousapient.processors.rtsp_event_producer import RTSPEventProducer
from moshousapient.processors.scene_anomaly_processor import SceneAnomalyProcessor
from moshousapient.streams.video_streamer import VideoStreamer


class RTSPPipeline:
    """
    代表一個完整的 RTSP 影像處理管線。

    它負責協調影像流讀取、AI 推論和行為分析等一系列處理器。
    """

    def __init__(
        self,
        camera_config: dict,
        model: YOLO,
        reid_model: YOLO,
        notifier: Any,
        rtsp_event_active_flag: Any,
    ):
        """
        初始化 RTSP 處理管線。

        :param camera_config: 攝影機的設定字典。
        :param model: 用於物件偵測的 YOLO 模型。
        :param reid_model: 用於特徵提取的 Re-ID 模型。
        :param notifier: 用於發送通知的服務實例。
        :param rtsp_event_active_flag: 跨程序共享的事件活躍標誌。
        """
        self.config = camera_config
        self.name = self.config.get("name", "RTSP-Pipeline-Default")
        self.notifier = notifier
        self.shared_state = {
            "person_detected": False,
            "tracked_objects": [],
            "scene_anomaly_detected": False,
            "scene_anomaly_type": None,
        }
        self.shared_state_lock = threading.Lock()
        self.video_streamer = VideoStreamer(self.config, Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT)
        is_target_mode = settings.VIDEO_FPS_MODE == "TARGET" and settings.TARGET_FPS > 0
        effective_fps = settings.TARGET_FPS if is_target_mode else self.video_streamer.source_fps
        logging.info(f"[{self.name}] 系統有效幀率設定為: {effective_fps:.2f} FPS (模式: {settings.VIDEO_FPS_MODE})")
        self.inference_queue = Queue(maxsize=2)
        self.anomaly_queue = Queue(maxsize=int(effective_fps * 2))
        buffer_size = int((Config.PRE_EVENT_SECONDS + 1.0) * effective_fps)
        self.event_queue = Queue(maxsize=buffer_size)
        self.video_streamer.add_output_queue(self.inference_queue)
        self.video_streamer.add_output_queue(self.anomaly_queue)
        self.video_streamer.add_output_queue(self.event_queue)

        self.processors = [
            InferenceProcessor(
                frame_queue=self.inference_queue,
                shared_state=self.shared_state,
                state_lock=self.shared_state_lock,
                model=model,
                reid_model=reid_model,
                tracker_factory=self._initialize_tracker,
                name=f"{self.name}-Inference",
            ),
            SceneAnomalyProcessor(
                frame_queue=self.anomaly_queue,
                shared_state=self.shared_state,
                state_lock=self.shared_state_lock,
                name=f"{self.name}-Anomaly",
            ),
            RTSPEventProducer(
                frame_queue=self.event_queue,
                shared_state=self.shared_state,
                state_lock=self.shared_state_lock,
                notifier=self.notifier,
                source_fps=effective_fps,  # <-- 注入有效幀率
                rtsp_event_active_flag=rtsp_event_active_flag,
                name=f"{self.name}-Event",
            ),
        ]
        logging.debug(f"[{self.name}] 已初始化。")

    def start(self):
        """啟動管線中的所有處理執行緒。"""
        logging.debug(f"[{self.name}] 正在啟動...")
        try:
            self.video_streamer.start()
            for processor in self.processors:
                processor.start()
            logging.info(f"[{self.name}] 所有處理執行緒已成功啟動。")
        except Exception as e:
            logging.error(f"[{self.name}] 啟動時發生錯誤: {e}", exc_info=True)
            self.stop()

    def stop(self):
        """安全地停止管線中的所有處理執行緒。"""
        logging.debug(f"[{self.name}] 正在關閉...")
        if self.video_streamer:
            self.video_streamer.stop()
        for processor in self.processors:
            processor.stop()
        logging.debug(f"[{self.name}] 已安全關閉。")

    def is_alive(self) -> bool:
        """檢查管線的核心影像流是否仍在運行。"""
        return self.video_streamer and self.video_streamer.is_alive()

    def _initialize_tracker(self):
        """根據設定檔初始化追蹤器物件。"""
        try:
            with open(Config.TRACKER_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f)
            tracker_args = SimpleNamespace(**cfg_dict)
            return BOTSORT(args=tracker_args)
        except Exception as e:
            logging.error(f"[{self.name}] 解析追蹤器設定檔時發生錯誤: {e}", exc_info=True)
            return None
