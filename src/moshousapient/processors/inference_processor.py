# src/moshousapient/processors/inference_processor.py
"""
定義了 InferenceProcessor，一個專門負責執行 AI 模型推論和物件追蹤的處理器。
"""

# 1. 標準庫導入
import logging
import time
from queue import Empty, Queue
from threading import Lock
from typing import Callable

# 2. 第三方庫導入
import numpy as np
from ultralytics import YOLO

# 3. 本專案導入
from moshousapient.processors.base_processor import BaseProcessor


class InferenceProcessor(BaseProcessor):
    """
    專門負責執行 AI 模型推論 (物件偵測、Re-ID) 和物件追蹤的處理器。
    """

    def __init__(
        self,
        frame_queue: Queue,
        shared_state: dict,
        state_lock: Lock,
        model: YOLO,
        reid_model: YOLO,
        tracker_factory: Callable,
        name: str = "InferenceProcessor",
    ):
        """
        初始化 InferenceProcessor。

        :param frame_queue: 包含待處理幀的輸入佇列。
        :param shared_state: 用於與其他處理器交換狀態的共享字典。
        :param state_lock: 用於保護 shared_state 的執行緒鎖。
        :param model: 用於物件偵測的 YOLO 模型。
        :param reid_model: 用於特徵提取的 Re-ID 模型。
        :param tracker_factory: 一個用於創建追蹤器實例的工廠函式。
        :param name: 處理器的名稱。
        """
        super().__init__(name)
        self.frame_queue = frame_queue
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.model = model
        self.reid_model = reid_model
        self.tracker_factory = tracker_factory
        self.tracker = self.tracker_factory()

    def _target_func(self):
        """
        主處理迴圈，持續從佇列獲取幀並執行推論和追蹤。
        """
        logging.info(f"[{self.name}] 處理器已啟動，使用 GPU 進行推論。")
        frame_counter = 0
        reid_interval = 5  # 每 5 幀提取一次 Re-ID 特徵

        while not self.stop_event.is_set():
            try:
                if self.stop_event.is_set() and self.frame_queue.empty():
                    break

                item = self.frame_queue.get(timeout=1)
                frame_counter += 1

                # 圖像縮放已在 VideoStreamer 中完成，這裡直接使用
                frame_for_inference = item["frame"]

                dets_results = self.model(frame_for_inference, device=0, verbose=False, classes=[0], conf=0.4)
                boxes_on_cpu = dets_results[0].boxes.cpu().numpy()
                tracks = self.tracker.update(boxes_on_cpu, frame_for_inference) if self.tracker else np.empty((0, 8))

                reid_features_map = {}
                if len(tracks) > 0 and (frame_counter % reid_interval == 0):
                    reid_features_map = self._extract_reid_features(tracks, frame_for_inference)

                with self.state_lock:
                    self.shared_state["person_detected"] = len(tracks) > 0
                    self.shared_state["tracked_objects"] = tracks
                    if reid_features_map:
                        self.shared_state["reid_features_map"] = reid_features_map

            except Empty:
                continue
            except Exception as e:
                logging.error(f"[{self.name}] 執行緒發生未預期的錯誤: {e}", exc_info=True)
                time.sleep(1)

        logging.info(f"[{self.name}] 處理器已停止。")

    def _extract_reid_features(self, tracks, frame) -> dict:
        """
        從追蹤到的物件中提取 Re-ID 特徵。

        :param tracks: 追蹤器輸出的結果。
        :param frame: 用於提取特徵的幀。
        :return: 一個字典，key 為 track_id，value 為特徵向量。
        """
        reid_features_map = {}
        track_ids = tracks[:, 4].astype(int)
        xyxy_coords = tracks[:, :4]

        person_crops, valid_track_ids = [], []
        for i, xyxy in enumerate(xyxy_coords):
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                person_crops.append(crop)
                valid_track_ids.append(track_ids[i])

        if person_crops:
            embeddings = self.reid_model.embed(person_crops, verbose=False)
            for i, track_id in enumerate(valid_track_ids):
                reid_features_map[track_id] = embeddings[i].cpu().numpy()
        return reid_features_map
