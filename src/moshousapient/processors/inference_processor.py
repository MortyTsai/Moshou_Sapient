# src/moshousapient/processors/inference_processor.py

import logging
import time
from queue import Queue, Empty
from threading import Lock
from typing import Callable
import numpy as np
import cv2
from ultralytics import YOLO
from shapely.geometry import Polygon, Point

from .base_processor import BaseProcessor
from ..config import Config
# 導入我們新的工具函式
from ..utils.geometry_utils import calculate_anchor_points


class InferenceProcessor(BaseProcessor):
    def __init__(self, frame_queue: Queue, shared_state: dict, state_lock: Lock,
                 model: YOLO, reid_model: YOLO, tracker_factory: Callable,
                 name: str = "InferenceProcessor"):
        super().__init__(name)
        self.frame_queue = frame_queue
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.model = model
        self.reid_model = reid_model
        self.tracker_factory = tracker_factory
        self.tracker = self.tracker_factory()

    def _target_func(self):
        logging.info(f"[{self.name}] 處理器已啟動, 使用 GPU 進行推論。")
        frame_counter = 0
        reid_interval = 5

        while not self.stop_event.is_set():
            try:
                if self.stop_event.is_set() and self.frame_queue.empty():
                    break

                with self.state_lock:
                    if self.shared_state.get('event_ended', False):
                        if self.tracker:
                            self.tracker = self.tracker_factory()
                        self.shared_state['event_ended'] = False
                        logging.info(f"[{self.name}] 偵測到事件結束, 已重新實例化追蹤器。")

                item = self.frame_queue.get(timeout=1)
                frame_counter += 1
                original_frame = item['frame']

                frame_low_res = cv2.resize(
                    original_frame,
                    (Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT),
                    interpolation=cv2.INTER_LINEAR
                )

                dets_results = self.model(frame_low_res, device=0, verbose=False, classes=[0], conf=0.4)

                boxes_on_cpu = dets_results[0].boxes.cpu().numpy()
                tracks = self.tracker.update(boxes_on_cpu, frame_low_res) if self.tracker else np.empty((0, 5))

                # 使用重構後的方法計算 ROI 狀態
                track_roi_status = self._calculate_roi_status(tracks)

                reid_features_map = {}
                if len(tracks) > 0 and (frame_counter % reid_interval == 0):
                    reid_features_map = self._extract_reid_features(tracks, frame_low_res)

                with self.state_lock:
                    self.shared_state['person_detected'] = len(tracks) > 0
                    self.shared_state['tracked_objects'] = tracks
                    self.shared_state['track_roi_status'] = track_roi_status
                    if reid_features_map:
                        self.shared_state['reid_features_map'] = reid_features_map

            except Empty:
                continue
            except Exception as e:
                logging.error(f"[{self.name}] 執行緒發生未預期的錯誤: {e}", exc_info=True)
                time.sleep(1)

        logging.info(f"[{self.name}] 處理器已停止。")

    @staticmethod
    def _calculate_roi_status(tracks) -> dict:
        """
        根據設定檔中定義的錨點策略，計算每個追蹤目標是否在 ROI 區域內。
        """
        if not Config.ROI_ENABLED or not Config.ROI_POLYGON_OBJECT:
            return {}

        # 優先使用 ROI 規則中覆寫的錨點，如果沒有，則使用全域錨點設定
        anchor_strategy = Config.ROI_SETTINGS.get('anchor_points', Config.ANCHOR_POINTS)

        roi_polygon = Config.ROI_POLYGON_OBJECT
        track_roi_status = {}

        for track in tracks:
            track_id = int(track[4])
            bbox = track[:4]
            is_in_roi = False

            # 呼叫工具函式計算所有錨點
            anchors = calculate_anchor_points(bbox, anchor_strategy)

            # 檢查是否有任何一個錨點滿足條件
            for anchor in anchors:
                if isinstance(anchor, Point):
                    # 如果錨點是 Point，使用 contains 判斷
                    if roi_polygon.contains(anchor):
                        is_in_roi = True
                        break  # 找到一個滿足條件的就足夠了
                elif isinstance(anchor, Polygon):
                    # 如果錨點是 Polygon (例如 'full_bbox')，使用 intersects 判斷
                    if roi_polygon.intersects(anchor):
                        is_in_roi = True
                        break

            track_roi_status[track_id] = is_in_roi

        return track_roi_status

    def _extract_reid_features(self, tracks, frame) -> dict:
        # ... (此方法保持不變) ...
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