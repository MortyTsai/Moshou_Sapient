# src/moshousapient/processors/event_processor.py
"""
此模組定義了 EventProcessor，負責處理高階行為分析（如 ROI、Tripwire）
和事件生命週期管理。它作為任務生產者，將待處理的
影片事件發送到任務佇列。
"""

# 1. 標準庫導入
import logging
import time
import pickle
import tempfile
import os
from collections import deque
from datetime import datetime
from queue import Empty, Queue
from threading import Lock
from typing import Union, Dict, cast, Tuple, Any

# 2. 第三方庫導入
import numpy as np
from shapely.geometry import Point

# 3. 本專案相對導入
from .base_processor import BaseProcessor
from ..config import Config
from ..services.task_queue_service import TaskQueueService
from ..utils.geometry_utils import calculate_anchor_points
from ..utils.behavior_utils import analyze_roi_status, analyze_tripwire_crossings

behavior_logger = logging.getLogger("BehaviorAnalysis")


class EventProcessor(BaseProcessor):
    """
    負責處理高階行為分析（如 ROI、Tripwire）和事件生命週期管理的處理器。
    它作為一個生產者，將偵測到的事件打包成任務並發送到 TaskQueueService。
    """

    def __init__(self, frame_queue: Queue, shared_state: dict, state_lock: Lock,
                 notifier, target_fps: float, name: str = "EventProcessor"):
        """
        初始化 EventProcessor。

        :param frame_queue: 包含幀數據的輸入佇列。
        :param shared_state: 用於與其他處理器交換狀態的共享字典。
        :param state_lock: 用於保護 shared_state 的執行緒鎖。
        :param notifier: 用於發送通知的通知器物件。
        :param target_fps: 目標處理幀率。
        :param name: 處理器的名稱。
        """
        super().__init__(name)
        self.frame_queue = frame_queue
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.notifier = notifier
        self.target_fps = target_fps
        self.task_queue = TaskQueueService()

        # --- 事件狀態管理 ---
        buffer_size = int((Config.PRE_EVENT_SECONDS + 1.0) * target_fps)
        self.frame_buffer = deque(maxlen=buffer_size)
        self.is_capturing_event = False
        self.event_start_time = 0.0
        self.last_event_end_time = 0.0
        self.last_person_seen_time = 0.0
        self.current_event_type: Union[str, None] = "person_detected"
        self.event_recording_frames: list = []

        # --- 行為分析狀態管理 ---
        self.dwell_time_trackers: Dict[int, Dict[str, Union[float, bool]]] = {}
        self.track_last_positions: Dict[Tuple[int, int], Point] = {}
        self.tripwire_alert_ids: set = set()

        # --- 幀率控制狀態 ---
        self.last_processed_time = 0.0
        self.time_accumulator = 0.0
        self.min_frame_interval = 1.0 / self.target_fps if Config.VIDEO_FPS_MODE == "TARGET" and self.target_fps > 0 else 0

    def _target_func(self):
        """
        主處理迴圈，持續從佇列中獲取幀並進行行為分析。
        這是執行緒執行的主要目標函式。
        """
        logging.info(f"[{self.name}] 處理器已啟動。")
        self.last_processed_time = time.time()

        while not self.stop_event.is_set():
            try:
                if self.stop_event.is_set() and self.frame_queue.empty():
                    break

                item = self.frame_queue.get(timeout=1)
                current_time = item['time']

                if self.min_frame_interval > 0:
                    delta_time = current_time - self.last_processed_time
                    self.last_processed_time = current_time
                    self.time_accumulator += delta_time

                    if self.time_accumulator < self.min_frame_interval:
                        continue
                    self.time_accumulator -= self.min_frame_interval

                with self.state_lock:
                    person_detected_now = self.shared_state.get('person_detected', False)
                    current_tracks_obj = self.shared_state.get('tracked_objects', [])

                current_tracks = np.array(current_tracks_obj)
                if current_tracks.size == 0:
                    current_tracks = np.empty((0, 8))

                track_roi_status_now = analyze_roi_status(
                    tracks=current_tracks,
                    roi_enabled=Config.ROI_ENABLED,
                    roi_polygon=Config.ROI_POLYGON_OBJECT,
                    roi_settings=Config.ROI_SETTINGS,
                    global_anchor_points=Config.ANCHOR_POINTS
                )

                crossed_ids_map, self.track_last_positions = analyze_tripwire_crossings(
                    tracks=current_tracks,
                    track_last_positions=self.track_last_positions,
                    tripwires_enabled=Config.TRIPWIRES_ENABLED,
                    tripwire_line_objects=Config.TRIPWIRE_LINE_OBJECTS,
                    global_anchor_points=Config.ANCHOR_POINTS
                )

                if crossed_ids_map:
                    self._handle_tripwire_logic(crossed_ids_map)

                self._handle_dwell_logic(track_roi_status_now, current_time)

                frame_data = self._prepare_frame_data(item, current_tracks, track_roi_status_now)
                self._update_event_state(person_detected_now, current_time, frame_data)

            except Empty:
                if self.is_capturing_event:
                    self._end_event(time.time(), "影像佇列為空")
                continue
            except Exception as e:
                logging.error(f"[{self.name}] 執行緒發生未預期的錯誤: {e}", exc_info=True)
                time.sleep(1)

        if self.is_capturing_event and self.event_recording_frames:
            self._end_event(time.time(), "系統關閉")

        logging.info(f"[{self.name}] 處理器已停止。")

    def _prepare_frame_data(self, item: dict, current_tracks: np.ndarray,
                            track_roi_status: Dict[int, bool]) -> Dict[str, Any]:
        """
        準備用於視覺化和儲存的單幀數據。

        :param item: 從佇列中獲取的原始項目。
        :param current_tracks: 當前的追蹤物件陣列。
        :param track_roi_status: 當前幀的 ROI 狀態字典。
        :return: 一個包含所有必要資訊的字典。
        """
        tracks_with_anchors = []
        alert_ids_snapshot = self.tripwire_alert_ids.copy()

        for track in current_tracks:
            track_id = int(track[4])
            bbox = track[:4]

            vis_anchor_strategy = Config.ANCHOR_POINTS
            if track_id in alert_ids_snapshot:
                for tripwire_obj in Config.TRIPWIRE_LINE_OBJECTS:
                    vis_anchor_strategy = tripwire_obj["config"].get('anchor_points', Config.ANCHOR_POINTS)
                    break
            elif track_roi_status.get(track_id, False):
                vis_anchor_strategy = Config.ROI_SETTINGS.get('anchor_points', Config.ANCHOR_POINTS)

            bbox_tuple = cast(Tuple[float, float, float, float], tuple(bbox))
            anchors = calculate_anchor_points(bbox_tuple, vis_anchor_strategy)
            anchor_coords = [list(anchor.coords)[0] for anchor in anchors if isinstance(anchor, Point)]

            is_in_roi = track_roi_status.get(track_id, False)
            has_crossed = track_id in alert_ids_snapshot

            tracks_with_anchors.append({
                'box_xyxy': bbox.tolist(),
                'track_id': track_id,
                'confidence': track[5] if len(track) > 5 else None,
                'anchors': anchor_coords,
                'is_in_roi': is_in_roi,
                'has_crossed_tripwire': has_crossed
            })

        return {
            'frame': item['frame'],
            'time': item['time'],
            'tracks': tracks_with_anchors,
            'track_roi_status': track_roi_status,
            'tripwire_alert_ids': alert_ids_snapshot,
        }

    def _update_event_state(self, person_detected_now: bool, current_time: float, frame_data: dict):
        """
        根據當前狀態更新事件的生命週期（開始、持續、結束）。

        :param person_detected_now: 當前幀是否偵測到人物。
        :param current_time: 當前幀的時間戳。
        :param frame_data: 準備好的當前幀數據。
        """
        if self.is_capturing_event:
            self.event_recording_frames.append(frame_data)

            post_event_elapsed = (current_time - self.last_person_seen_time) > Config.POST_EVENT_SECONDS
            max_duration_reached = (current_time - self.event_start_time) > Config.MAX_EVENT_DURATION

            if not person_detected_now and post_event_elapsed:
                self._end_event(current_time, "人物消失")
            elif max_duration_reached:
                self._end_event(current_time, "超過最大錄影時長", is_segment=True)
        else:
            self.frame_buffer.append(frame_data)

        if person_detected_now:
            self.last_person_seen_time = current_time
            is_in_cooldown = (current_time - self.last_event_end_time) <= Config.COOLDOWN_PERIOD
            if not self.is_capturing_event and not is_in_cooldown:
                self._start_event(current_time)

    def _start_event(self, current_time: float):
        """
        開始一個新的事件錄製。

        :param current_time: 事件開始的時間戳。
        """
        self.is_capturing_event = True
        self.event_start_time = current_time
        self.current_event_type = "person_detected"
        self.event_recording_frames = list(self.frame_buffer)
        logging.info(f">>> [事件] 偵測到 '{self.current_event_type}' 事件! 開始錄製...")

    def _end_event(self, current_time: float, reason: str, is_segment: bool = False):
        """
        結束當前事件錄製或進行分段，並將任務發送到佇列。

        :param current_time: 事件結束的時間戳。
        :param reason: 事件結束的原因。
        :param is_segment: 是否為事件分段。
        """
        logging.info(f"[事件] 事件結束 ({reason})。")

        if self.event_recording_frames:
            self._dispatch_video_task()

        self.last_event_end_time = current_time
        if is_segment:
            logging.info(">>> [事件] 進行事件分段，準備錄製下一段...")
            overlap_frames = int(Config.PRE_EVENT_SECONDS * self.target_fps)
            self.event_recording_frames = self.event_recording_frames[-overlap_frames:]
            self.event_start_time = current_time
            self.current_event_type = "person_detected"
        else:
            self.is_capturing_event = False
            self.current_event_type = None
            self.event_recording_frames.clear()
            self.dwell_time_trackers.clear()
            self.tripwire_alert_ids.clear()
            self.track_last_positions.clear()

    def _dispatch_video_task(self):
        """
        將捕獲到的幀數據寫入臨時檔案，並將任務發送到任務佇列。
        """
        temp_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', dir=str(Config.CAPTURES_DIR)) as temp_f:
                pickle.dump(self.event_recording_frames, temp_f)
                temp_file_path = temp_f.name

            now = datetime.fromtimestamp(self.event_start_time)
            filename = f"{self.current_event_type}_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = os.path.join(Config.CAPTURES_DIR, filename)

            rendering_config = {
                "roi_enabled": Config.ROI_ENABLED,
                "roi_polygon": Config.ROI_POLYGON_OBJECT,
                "tripwires_enabled": Config.TRIPWIRES_ENABLED,
                "tripwire_line_objects": Config.TRIPWIRE_LINE_OBJECTS,
            }

            payload = {
                "data_path": temp_file_path,
                "output_path": output_path,
                "event_type": self.current_event_type,
                "source_meta": {},
                "event_start_frame_index": -1,
                "rendering_config": rendering_config
            }

            payload_bytes = pickle.dumps(payload)

            task_id = self.task_queue.add_task(payload_bytes)
            if task_id:
                logging.info(f"已成功將事件 '{self.current_event_type}' 作為任務 ID {task_id} 發送到佇列。")
            else:
                logging.error("將事件發送到任務佇列失敗，正在清理臨時檔案...")
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        except (IOError, pickle.PicklingError) as e:
            logging.error(f"建立事件任務時發生錯誤: {e}", exc_info=True)
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _set_event_type(self, new_type: str):
        """
        根據優先級提升當前事件的類型。

        :param new_type: 新的事件類型。
        """
        priority_map = {'tripwire_alert': 2, 'dwell_alert': 1, 'person_detected': 0}
        current_priority = priority_map.get(self.current_event_type, -1)
        new_priority = priority_map.get(new_type, -1)

        if new_priority > current_priority:
            if self.is_capturing_event:
                logging.info(f">>> [事件升級] '{self.current_event_type}' 事件已升級為 '{new_type}'")
            self.current_event_type = new_type

    def _handle_dwell_logic(self, track_roi_status: Dict[int, bool], current_time: float):
        """
        處理所有與 ROI 區域停留相關的邏輯。

        :param track_roi_status: 當前幀的 ROI 狀態字典。
        :param current_time: 當前幀的時間戳。
        """
        if not Config.ROI_ENABLED:
            return

        current_tracked_ids = set(track_roi_status.keys())
        for track_id, is_in_roi in track_roi_status.items():
            if is_in_roi:
                if track_id not in self.dwell_time_trackers:
                    self.dwell_time_trackers[track_id] = {'start_time': current_time, 'alerted': False}
                else:
                    tracker_info = self.dwell_time_trackers[track_id]
                    if not tracker_info['alerted']:
                        dwell_duration = current_time - tracker_info['start_time']
                        dwell_threshold = Config.ROI_SETTINGS.get('dwell_time_threshold', 3.0)
                        if dwell_duration > dwell_threshold:
                            behavior_logger.warning(
                                f"--- [停留警報] --- 目標 ID: {track_id} 在 ROI 區域停留已超過 {dwell_threshold} 秒!"
                            )
                            self._set_event_type("dwell_alert")
                            tracker_info['alerted'] = True
            else:
                if track_id in self.dwell_time_trackers:
                    del self.dwell_time_trackers[track_id]

        disappeared_ids = set(self.dwell_time_trackers.keys()) - current_tracked_ids
        for track_id in disappeared_ids:
            del self.dwell_time_trackers[track_id]

    def _handle_tripwire_logic(self, crossed_ids_map: Dict[int, bool]):
        """
        處理所有與警戒線穿越相關的邏輯。

        :param crossed_ids_map: 包含已穿越目標 ID 的字典。
        """
        newly_crossed_ids = set(crossed_ids_map.keys()) - self.tripwire_alert_ids
        if newly_crossed_ids:
            self._set_event_type("tripwire_alert")
            for track_id in newly_crossed_ids:
                self.tripwire_alert_ids.add(track_id)
                behavior_logger.warning(f"--- [方向性警報] --- 目標 ID: {track_id} 觸發了警戒線。")