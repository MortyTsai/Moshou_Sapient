import logging
import pickle
import subprocess
import sys
import tempfile
import time
from collections import deque
from datetime import datetime
from queue import Empty, Queue
from threading import Lock
from typing import Union

from shapely.geometry import Point, LineString, Polygon

from .base_processor import BaseProcessor
from ..config import Config
from ..utils.geometry_utils import calculate_anchor_points, get_point_side_of_line

behavior_logger = logging.getLogger("BehaviorAnalysis")


class EventProcessor(BaseProcessor):
    """
    負責處理高階行為分析（如 ROI、Tripwire）和事件生命週期管理的處理器。
    """

    def __init__(self, frame_queue: Queue, shared_state: dict, state_lock: Lock,
                 notifier, target_fps: float, name: str = "EventProcessor"):
        """初始化 EventProcessor。"""
        super().__init__(name)
        self.frame_queue = frame_queue
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.notifier = notifier
        self.target_fps = target_fps

        buffer_size = int((Config.PRE_EVENT_SECONDS + 1.0) * target_fps)
        self.frame_buffer = deque(maxlen=buffer_size)
        self.is_capturing_event = False
        self.event_start_time = 0
        self.last_event_end_time = 0
        self.last_person_seen_time = 0
        self.current_event_type: Union[str, None] = "person_detected"
        self.event_recording_frames = []

        self.dwell_time_trackers = {}
        self.track_last_positions = {}
        self.tripwire_alert_ids = set()

    def _calculate_roi_status(self, tracks) -> dict:
        """
        根據設定檔中定義的錨點策略，計算每個追蹤目標是否在 ROI 區域內。
        此函式會優先使用 ROI 規則中定義的 'anchor_points'。
        """
        if not Config.ROI_ENABLED or not Config.ROI_POLYGON_OBJECT:
            return {}

        anchor_strategy = Config.ROI_SETTINGS.get('anchor_points', Config.ANCHOR_POINTS)
        roi_polygon = Config.ROI_POLYGON_OBJECT
        track_roi_status = {}

        for track in tracks:
            track_id = int(track[4])
            bbox = track[:4]
            is_in_roi = False

            anchors = calculate_anchor_points(bbox, anchor_strategy)

            for anchor in anchors:
                if isinstance(anchor, Point):
                    if roi_polygon.contains(anchor):
                        is_in_roi = True
                        break
                elif isinstance(anchor, Polygon):
                    if roi_polygon.intersects(anchor):
                        is_in_roi = True
                        break

            track_roi_status[track_id] = is_in_roi

        return track_roi_status

    def _target_func(self):
        """主處理迴圈，持續從佇列中獲取幀並進行行為分析。"""
        logging.info(f"[{self.name}] 處理器已啟動。")
        while not self.stop_event.is_set():
            try:
                if self.stop_event.is_set() and self.frame_queue.empty():
                    break

                item = self.frame_queue.get(timeout=1)
                current_time = item['time']

                with self.state_lock:
                    current_tracks = self.shared_state.get('tracked_objects', [])
                    person_detected_now = self.shared_state.get('person_detected', False)

                track_roi_status_now = self._calculate_roi_status(current_tracks)

                self._handle_tripwire_logic(current_tracks)
                self._handle_dwell_logic(track_roi_status_now, current_time)
                alert_ids_snapshot = self.tripwire_alert_ids.copy()

                tracks_with_anchors = []
                for track in current_tracks:
                    bbox = track[:4]
                    anchor_strategy = Config.ANCHOR_POINTS  # 視覺化錨點統一使用全域設定
                    anchors = calculate_anchor_points(bbox, anchor_strategy)
                    anchor_coords = [list(anchor.coords)[0] for anchor in anchors if isinstance(anchor, Point)]

                    track_info = {
                        'box_xyxy': bbox.tolist(), 'track_id': int(track[4]),
                        'confidence': track[5] if len(track) > 5 else None,
                        'anchors': anchor_coords
                    }
                    tracks_with_anchors.append(track_info)

                frame_data = {
                    'frame': item['frame'], 'time': current_time,
                    'tracks': tracks_with_anchors,
                    'track_roi_status': track_roi_status_now,
                    'tripwire_alert_ids': alert_ids_snapshot
                }

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

    def _update_event_state(self, person_detected_now: bool, current_time: float, frame_data: dict):
        """根據當前狀態更新事件的生命週期（開始、持續、結束）。"""
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
        """開始一個新的事件錄製。"""
        self.is_capturing_event = True
        self.event_start_time = current_time
        self.current_event_type = "person_detected"
        self.event_recording_frames = list(self.frame_buffer)
        logging.info(f">>> [事件] 偵測到 '{self.current_event_type}' 事件! 開始錄製...")

    def _end_event(self, current_time: float, reason: str, is_segment: bool = False):
        """結束當前事件錄製或進行分段。"""
        logging.info(f"[事件] 事件結束 ({reason})。")
        event_metadata = {
            "start_time": self.event_start_time, "end_time": current_time,
            "event_type": self.current_event_type
        }
        if self.event_recording_frames:
            self._start_isolated_video_processor(event_metadata, self.event_recording_frames)

        self.last_event_end_time = current_time
        if is_segment:
            logging.info(">>> [事件] 進行事件分段, 準備錄製下一段...")
            overlap_frames = int(Config.PRE_EVENT_SECONDS * self.target_fps)
            self.event_recording_frames = self.event_recording_frames[-overlap_frames:]
            self.event_start_time = current_time
        else:
            self.is_capturing_event = False
            self.current_event_type = None
            self.event_recording_frames = []
            self.dwell_time_trackers.clear()
            self.tripwire_alert_ids.clear()
            self.track_last_positions.clear()

    @staticmethod
    def _start_isolated_video_processor(event_metadata: dict, frames_data: list):
        """啟動一個獨立的子程序來處理影片的繪製和編碼。"""
        logging.info(f"正在為事件 '{event_metadata['event_type']}' 啟動獨立的影片處理服務...")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl',
                                             dir=str(Config.CAPTURES_DIR),
                                             mode='wb') as temp_f:
                pickle.dump(frames_data, temp_f)  # type: ignore
                temp_file_path = temp_f.name

            now = datetime.fromtimestamp(event_metadata['start_time'])
            filename = f"{event_metadata['event_type']}_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = str(Config.CAPTURES_DIR / filename)

            command = [
                sys.executable, "-m", "moshousapient.services.isolated_video_processor",
                "--input-data-path", temp_file_path,
                "--output-path", output_path,
                "--event-type", event_metadata.get('event_type') or "unknown"
            ]
            subprocess.Popen(command)
            logging.info(f"影片處理服務已成功啟動，輸出目標: {output_path}")
        except Exception as e:
            logging.error(f"啟動獨立影片處理服務失敗: {e}", exc_info=True)

    def _set_event_type(self, new_type: str):
        """根據優先級提升當前事件的類型。"""
        priority_map = {'tripwire_alert': 2, 'dwell_alert': 1, 'person_detected': 0}
        current_priority = priority_map.get(self.current_event_type, -1)
        new_priority = priority_map.get(new_type, -1)
        if new_priority > current_priority:
            if self.is_capturing_event:
                logging.info(f">>> [事件升級] '{self.current_event_type}' 事件已升級為 '{new_type}'")
            self.current_event_type = new_type

    def _handle_tripwire_logic(self, current_tracks):
        """處理所有與警戒線穿越相關的邏輯。"""
        if not Config.TRIPWIRES_ENABLED:
            return

        for track in current_tracks:
            track_id = int(track[4])
            bbox = track[:4]
            for tripwire_obj in Config.TRIPWIRE_LINE_OBJECTS:
                tripwire_line, alert_direction, tripwire_config = \
                    tripwire_obj["line"], tripwire_obj["direction"], tripwire_obj["config"]

                anchor_strategy = tripwire_config.get('anchor_points', Config.ANCHOR_POINTS)
                current_anchors = calculate_anchor_points(bbox, anchor_strategy)

                for i, current_anchor in enumerate(current_anchors):
                    if not isinstance(current_anchor, Point): continue
                    anchor_key = (track_id, i)
                    last_position = self.track_last_positions.get(anchor_key)

                    if last_position and last_position != current_anchor:
                        movement_line = LineString([last_position, current_anchor])
                        if movement_line.intersects(tripwire_line):
                            p1, p2 = tripwire_line.coords
                            side_before = get_point_side_of_line(last_position, Point(p1), Point(p2))
                            side_after = get_point_side_of_line(current_anchor, Point(p1), Point(p2))

                            if side_before != 0 and side_after != 0 and side_before != side_after:
                                crossed_to_right = side_before == -1 and side_after == 1
                                crossed_to_left = side_before == 1 and side_after == -1
                                should_alert = (alert_direction == "both" or
                                                (alert_direction == "cross_to_right" and crossed_to_right) or
                                                (alert_direction == "cross_to_left" and crossed_to_left))
                                if should_alert:
                                    if track_id not in self.tripwire_alert_ids:
                                        behavior_logger.warning(
                                            f"--- [方向性警報] --- 目標 ID: {track_id} 觸發了警戒線: "
                                            f"{tripwire_config.get('name', '未命名')}")
                                        self.tripwire_alert_ids.add(track_id)
                                        self._set_event_type("tripwire_alert")
                                    break
                    self.track_last_positions[anchor_key] = current_anchor
                else:
                    continue
                break

    def _handle_dwell_logic(self, track_roi_status, current_time):
        """處理所有與 ROI 區域停留相關的邏輯。"""
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
                                f"--- [停留警報] --- 目標 ID: {track_id} 在 ROI 區域停留已超過 {dwell_threshold} 秒!")
                            self._set_event_type("dwell_alert")
                            tracker_info['alerted'] = True
            else:
                if track_id in self.dwell_time_trackers:
                    del self.dwell_time_trackers[track_id]

        disappeared_ids = set(self.dwell_time_trackers.keys()) - current_tracked_ids
        for track_id in disappeared_ids:
            del self.dwell_time_trackers[track_id]