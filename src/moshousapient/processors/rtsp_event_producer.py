# src/moshousapient/processors/rtsp_event_producer.py
"""
定義了 RTSPEventProducer，一個作為生產者的處理器。

它負責處理高階行為分析 (如 ROI、Tripwire) 和事件生命週期管理，
並將一個完整的、連續的活動打包成單一任務，發送到任務佇列。
"""

# 1. 標準庫導入
import logging
import os
import pickle
import tempfile
import time
from collections import deque
from queue import Empty, Queue
from threading import Lock
from typing import Any, Dict, Tuple, Union, cast

# 2. 第三方庫導入
import numpy as np
from shapely.geometry import Point

# 3. 本專案導入
from moshousapient.configs.behavior_config import Config
from moshousapient.processors.base_processor import BaseProcessor
from moshousapient.services.task_queue_service import TaskQueueService
from moshousapient.utils.behavior_analysis_utils import (
    analyze_roi_status,
    analyze_tripwire_crossings,
)
from moshousapient.utils.geometry_utils import calculate_anchor_points

behavior_logger = logging.getLogger("BehaviorAnalysis")


class RTSPEventProducer(BaseProcessor):
    """
    負責處理高階行為分析和事件生命週期管理的處理器。
    """

    def __init__(
        self,
        frame_queue: Queue,
        shared_state: dict,
        state_lock: Lock,
        notifier: Any,
        target_fps: float,
        rtsp_event_active_flag: Any,
        name: str = "RTSPEventProducer",
    ):
        """
        初始化 RTSPEventProducer。
        """
        super().__init__(name)
        self.frame_queue = frame_queue
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.notifier = notifier
        self.target_fps = target_fps
        self.task_queue = TaskQueueService()
        self.rtsp_event_active_flag = rtsp_event_active_flag

        buffer_size = int((Config.PRE_EVENT_SECONDS + 1.0) * target_fps)
        self.frame_buffer = deque(maxlen=buffer_size)
        self.is_capturing_event = False
        self.event_start_time = 0.0
        self.last_event_end_time = 0.0
        self.last_person_seen_time = 0.0
        self.current_event_type: Union[str, None] = "person_detected"
        self.event_recording_frames: list = []

        self.dwell_time_trackers: Dict[int, Dict[str, Union[float, bool]]] = {}
        self.track_last_positions: Dict[Tuple[int, int], Point] = {}
        self.tripwire_alert_ids: set = set()

        self.last_processed_time = 0.0
        self.time_accumulator = 0.0
        self.min_frame_interval = (
            1.0 / self.target_fps if Config.VIDEO_FPS_MODE == "TARGET" and self.target_fps > 0 else 0
        )
        logging.debug(f"[{self.name}] 初始化完成。")

    def _start_event(self, current_time: float):
        """開始一個新的事件錄製。"""
        self.is_capturing_event = True
        self.event_start_time = current_time
        self.current_event_type = "person_detected"
        self.event_recording_frames = list(self.frame_buffer)
        self.rtsp_event_active_flag.value = True
        logging.info(f"偵測到事件 '{self.current_event_type}'! 開始錄製...")

    def _end_event(self, current_time: float, reason: str):
        """結束當前事件錄製，並將完整的事件打包發送到佇列。"""
        logging.debug(f"事件結束 ({reason})。")
        if self.event_recording_frames:
            self._dispatch_video_task()

        self.last_event_end_time = current_time
        self.is_capturing_event = False
        self.current_event_type = None
        self.event_recording_frames.clear()
        self.dwell_time_trackers.clear()
        self.tripwire_alert_ids.clear()
        self.track_last_positions.clear()
        self.rtsp_event_active_flag.value = False

    def _process_frame(self, item: Dict[str, Any], current_time: float):
        """處理單一幀的核心邏輯。"""
        with self.state_lock:
            person_detected_now = self.shared_state.get("person_detected", False)
            current_tracks_obj = self.shared_state.get("tracked_objects", [])

        current_tracks = np.array(current_tracks_obj)
        if current_tracks.size == 0:
            current_tracks = np.empty((0, 8))

        track_roi_status_now = analyze_roi_status(
            tracks=current_tracks,
            roi_enabled=Config.ROI_ENABLED,
            roi_polygon=Config.ROI_POLYGON_OBJECT,
            roi_settings=Config.ROI_SETTINGS,
            global_anchor_points=Config.ANCHOR_POINTS,
        )
        crossed_ids_map, self.track_last_positions = analyze_tripwire_crossings(
            tracks=current_tracks,
            track_last_positions=self.track_last_positions,
            tripwires_enabled=Config.TRIPWIRES_ENABLED,
            tripwire_line_objects=Config.TRIPWIRE_LINE_OBJECTS,
            global_anchor_points=Config.ANCHOR_POINTS,
        )

        if crossed_ids_map:
            self._handle_tripwire_logic(crossed_ids_map)

        self._handle_dwell_logic(track_roi_status_now, current_time)

        frame_data = self._prepare_frame_data(item, current_tracks, track_roi_status_now)
        self._update_event_state(person_detected_now, current_time, frame_data)

    def _target_func(self):
        """主處理迴圈，持續從佇列中獲取幀並進行行為分析。"""
        logging.debug(f"[{self.name}] 處理器已啟動。")
        self.last_processed_time = time.time()

        while not self.stop_event.is_set():
            try:
                if self.stop_event.is_set() and self.frame_queue.empty():
                    break

                item = self.frame_queue.get(timeout=1)
                current_time = item["time"]

                if self.min_frame_interval > 0:
                    delta_time = current_time - self.last_processed_time
                    self.last_processed_time = current_time
                    self.time_accumulator += delta_time
                    if self.time_accumulator < self.min_frame_interval:
                        continue
                    self.time_accumulator -= self.min_frame_interval

                self._process_frame(item, current_time)

            except Empty:
                if self.is_capturing_event:
                    self._end_event(time.time(), "影像佇列為空")
                continue
            except Exception:
                logging.exception(f"[{self.name}] 執行緒發生未預期的錯誤")
                time.sleep(1)

        if self.is_capturing_event and self.event_recording_frames:
            self._end_event(time.time(), "系統關閉")

        if self.rtsp_event_active_flag.value:
            self.rtsp_event_active_flag.value = False

        logging.debug(f"[{self.name}] 處理器已停止。")

    def _prepare_frame_data(
        self, item: dict, current_tracks: np.ndarray, track_roi_status: Dict[int, bool]
    ) -> Dict[str, Any]:
        """準備用於視覺化和儲存的單幀數據。"""
        tracks_with_anchors = []
        alert_ids_snapshot = self.tripwire_alert_ids.copy()
        for track in current_tracks:
            track_id = int(track[4])
            bbox = track[:4]

            vis_anchor_strategy = Config.ANCHOR_POINTS
            if track_id in alert_ids_snapshot:
                for tripwire_obj in Config.TRIPWIRE_LINE_OBJECTS:
                    vis_anchor_strategy = tripwire_obj["config"].get("anchor_points", Config.ANCHOR_POINTS)
                    break
            elif track_roi_status.get(track_id, False):
                vis_anchor_strategy = Config.ROI_SETTINGS.get("anchor_points", Config.ANCHOR_POINTS)

            bbox_tuple = cast(Tuple[float, float, float, float], tuple(bbox))
            anchors = calculate_anchor_points(bbox_tuple, vis_anchor_strategy)
            anchor_coords = [next(iter(anchor.coords)) for anchor in anchors if isinstance(anchor, Point)]

            tracks_with_anchors.append(
                {
                    "box_xyxy": bbox.tolist(),
                    "track_id": track_id,
                    "confidence": track[5] if len(track) > 5 else None,
                    "anchors": anchor_coords,
                    "is_in_roi": track_roi_status.get(track_id, False),
                    "has_crossed_tripwire": track_id in alert_ids_snapshot,
                }
            )

        return {
            "frame": item["frame"],
            "time": item["time"],
            "tracks": tracks_with_anchors,
            "track_roi_status": track_roi_status,
            "tripwire_alert_ids": alert_ids_snapshot,
        }

    def _update_event_state(self, person_detected_now: bool, current_time: float, frame_data: dict):
        """根據當前狀態更新事件的生命週期（開始、持續、結束）。"""
        if person_detected_now:
            self.last_person_seen_time = current_time
            is_in_cooldown = (current_time - self.last_event_end_time) <= Config.COOLDOWN_PERIOD
            if not self.is_capturing_event and not is_in_cooldown:
                self._start_event(current_time)

        if self.is_capturing_event:
            self.event_recording_frames.append(frame_data)
            post_event_elapsed = (current_time - self.last_person_seen_time) > Config.POST_EVENT_SECONDS
            if not person_detected_now and post_event_elapsed:
                self._end_event(current_time, "人物消失")
            else:
                self.frame_buffer.append(frame_data)

    def _dispatch_video_task(self):
        """將捕獲到的幀數據寫入臨時檔案，並將任務發送到任務佇列。"""
        temp_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=str(Config.CAPTURES_DIR)) as temp_f:
                pickle.dump(self.event_recording_frames, temp_f)
                temp_file_path = temp_f.name

            payload = {
                "data_path": temp_file_path,
                "event_type": self.current_event_type,
                "source_meta": {},  # RTSP 模式下，元數據由消費者端處理
            }
            payload_bytes = pickle.dumps(payload)
            task_id = self.task_queue.add_task(payload_bytes)

            if task_id:
                logging.info(f"已成功將事件 '{self.current_event_type}' 作為任務 ID {task_id} 發送到佇列。")
            else:
                logging.error(f"將事件 '{self.current_event_type}' 發送到任務佇列失敗。")

        except (IOError, pickle.PicklingError):
            logging.exception("建立事件任務時發生錯誤")
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _set_event_type(self, new_type: str):
        """根據優先級提升當前事件的類型。"""
        priority_map = {"tripwire_alert": 2, "dwell_alert": 1, "person_detected": 0}
        current_priority = priority_map.get(self.current_event_type, -1)
        new_priority = priority_map.get(new_type, -1)

        if new_priority > current_priority:
            if self.is_capturing_event:
                logging.info(f"事件升級: '{self.current_event_type}' -> '{new_type}'")
            self.current_event_type = new_type

    def _handle_dwell_logic(self, track_roi_status: Dict[int, bool], current_time: float):
        """處理所有與 ROI 區域停留相關的邏輯。"""
        if not Config.ROI_ENABLED:
            return

        current_tracked_ids = set(track_roi_status.keys())
        for track_id, is_in_roi in track_roi_status.items():
            if is_in_roi:
                if track_id not in self.dwell_time_trackers:
                    self.dwell_time_trackers[track_id] = {
                        "start_time": current_time,
                        "alerted": False,
                    }
                else:
                    tracker_info = self.dwell_time_trackers[track_id]
                    if not tracker_info["alerted"]:
                        dwell_duration = current_time - tracker_info["start_time"]
                        dwell_threshold = Config.ROI_SETTINGS.get("dwell_time_threshold", 3.0)
                        if dwell_duration > dwell_threshold:
                            behavior_logger.warning(
                                f"--- [停留警報] --- 目標 ID: {track_id} 在 ROI 區域停留已超過 {dwell_threshold} 秒!"
                            )
                            self._set_event_type("dwell_alert")
                            tracker_info["alerted"] = True
            elif track_id in self.dwell_time_trackers:
                del self.dwell_time_trackers[track_id]

        disappeared_ids = set(self.dwell_time_trackers.keys()) - current_tracked_ids
        for track_id in disappeared_ids:
            del self.dwell_time_trackers[track_id]

    def _handle_tripwire_logic(self, crossed_ids_map: Dict[int, bool]):
        """處理所有與警戒線穿越相關的邏輯。"""
        newly_crossed_ids = set(crossed_ids_map.keys()) - self.tripwire_alert_ids
        if newly_crossed_ids:
            self._set_event_type("tripwire_alert")
            for track_id in newly_crossed_ids:
                self.tripwire_alert_ids.add(track_id)
                behavior_logger.warning(f"--- [方向性警報] --- 目標 ID: {track_id} 觸發了警戒線。")
