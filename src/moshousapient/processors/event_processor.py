# src/moshousapient/processors/event_processor.py (Definitive Final Cleaned Version 2)
import logging, time, sys, subprocess, pickle, tempfile
from queue import Queue, Empty
from threading import Lock
from collections import deque
from typing import Union
from datetime import datetime
from ..config import Config
from .base_processor import BaseProcessor
from ..utils.geometry_utils import get_point_side_of_line, calculate_anchor_points
from shapely.geometry import Point, LineString

behavior_logger = logging.getLogger("BehaviorAnalysis")


class EventProcessor(BaseProcessor):
    def __init__(self, frame_queue: Queue, shared_state: dict, state_lock: Lock,
                 notifier, target_fps: float, name: str = "EventProcessor"):
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
        self.current_event_type: Union[str, None] = None
        self.event_recording_frames = []
        self.event_recording_features = []
        self.dwell_time_trackers = {}
        self.track_last_positions = {}
        self.tripwire_alert_ids = set()

    def _target_func(self):
        # ... (此方法無變化) ...
        logging.info(f"[{self.name}] 處理器已啟動。")
        while not self.stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=1)
                current_time = item['time']
                with self.state_lock:
                    current_tracks = self.shared_state.get('tracked_objects', [])
                    person_detected_now = self.shared_state.get('person_detected', False)
                    track_roi_status_now = self.shared_state.get('track_roi_status', {})
                    reid_features_to_add = self.shared_state.get('reid_features_map', {})

                self._handle_tripwire_logic(current_tracks)
                self._handle_dwell_logic(track_roi_status_now, current_time)
                alert_ids_snapshot = self.tripwire_alert_ids.copy()
                frame_data = {
                    'frame': item['frame'], 'time': current_time,
                    'tracks': current_tracks,
                    'track_roi_status': track_roi_status_now,
                    'tripwire_alert_ids': alert_ids_snapshot
                }
                if self.is_capturing_event:
                    self.event_recording_frames.append(frame_data)
                    if reid_features_to_add:
                        self.event_recording_features.extend(reid_features_to_add.values())
                else:
                    self.frame_buffer.append(frame_data)

                if person_detected_now:
                    self.last_person_seen_time = current_time

                if not self.is_capturing_event:
                    is_in_cooldown = current_time - self.last_event_end_time <= Config.COOLDOWN_PERIOD
                    if person_detected_now and not is_in_cooldown:
                        self._start_event(current_time)
                else:
                    post_event_elapsed = current_time - self.last_person_seen_time > Config.POST_EVENT_SECONDS
                    max_duration_reached = current_time - self.event_start_time > Config.MAX_EVENT_DURATION
                    if not person_detected_now and post_event_elapsed:
                        self._end_event(current_time, "人物消失")
                    elif max_duration_reached:
                        self._end_event(current_time, "超過最大錄影時長", is_segment=True)

            except Empty:
                if self.is_capturing_event: self._end_event(time.time(), "影像佇列為空")
                continue
            except Exception as e:
                logging.error(f"[{self.name}] 執行緒發生未預期的錯誤: {e}", exc_info=True)
                time.sleep(1)
        if self.is_capturing_event and self.event_recording_frames:
            self._end_event(time.time(), "系統關閉")
        logging.info(f"[{self.name}] 處理器已停止。")

    def _start_event(self, current_time):
        # ... (此方法無變化) ...
        self.is_capturing_event = True
        self.event_start_time = current_time
        self.current_event_type = "person_detected"
        self.event_recording_frames = list(self.frame_buffer)
        self.event_recording_features = []
        logging.info(f">>> [事件] 偵測到 '{self.current_event_type}' 事件! 開始錄製...")

    def _end_event(self, current_time, reason: str, is_segment: bool = False):
        # ... (此方法無變化) ...
        logging.info(f"[事件] 事件結束 ({reason})。")
        event_metadata = {
            "start_time": self.event_start_time,
            "end_time": current_time,
            "event_type": self.current_event_type,
            "features": self.event_recording_features
        }
        if self.event_recording_frames:
            self._start_isolated_video_processor(event_metadata, self.event_recording_frames)

        self.last_event_end_time = current_time
        if is_segment:
            logging.info(">>> [事件] 進行事件分段，準備錄製下一段...")
            overlap_frames = int(Config.PRE_EVENT_SECONDS * self.target_fps)
            self.event_recording_frames = self.event_recording_frames[-overlap_frames:]
            self.event_recording_features = []
            self.event_start_time = current_time
        else:
            self.is_capturing_event = False
            self.current_event_type = None
            self.event_recording_frames = []
            self.event_recording_features = []
            self.dwell_time_trackers.clear()
            self.tripwire_alert_ids.clear()
            self.track_last_positions.clear()

    @staticmethod
    def _start_isolated_video_processor(event_metadata: dict, frames_data: list):
        # ... (此方法無變化) ...
        logging.info(f"正在為事件 '{event_metadata['event_type']}' 啟動獨立的影片處理服務...")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', dir=str(Config.CAPTURES_DIR),
                                             mode='wb') as temp_f:
                pickle.dump(frames_data, temp_f)
                temp_file_path = temp_f.name

            now = datetime.fromtimestamp(event_metadata['start_time'])
            filename = f"{event_metadata['event_type']}_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = str(Config.CAPTURES_DIR / filename)

            command = [
                sys.executable, "-m", "moshousapient.services.isolated_video_processor",
                "--input-data-path", temp_file_path,
                "--output-path", output_path,
                "--event-type", event_metadata['event_type'] or "unknown"
            ]

            subprocess.Popen(command)
            logging.info(f"影片處理服務已成功啟動，輸出目標: {output_path}")
        except Exception as e:
            logging.error(f"啟動獨立影片處理服務失敗: {e}", exc_info=True)

    def _set_event_type(self, new_type: str):
        # ... (此方法無變化) ...
        priority_map = {"tripwire_alert": 2, "dwell_alert": 1, "person_detected": 0}
        current_priority = priority_map.get(self.current_event_type, -1)
        new_priority = priority_map.get(new_type, -1)
        if new_priority > current_priority:
            if self.is_capturing_event:
                logging.info(f">>> [事件升級] '{self.current_event_type}' 事件已升級為 '{new_type}'")
            self.current_event_type = new_type

    def _handle_tripwire_logic(self, current_tracks):
        # ... (此方法無變化) ...
        if not Config.TRIPWIRES_ENABLED: return
        for track in current_tracks:
            track_id = int(track[4])
            bbox = track[:4]
            for tripwire_obj in Config.TRIPWIRE_LINE_OBJECTS:
                tripwire_line, alert_direction, tripwire_config = tripwire_obj["line"], tripwire_obj["direction"], \
                tripwire_obj["config"]
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
                                crossed_to_right = side_before == 1 and side_after == -1
                                crossed_to_left = side_before == -1 and side_after == 1
                                should_alert = (alert_direction == "both" or (
                                            alert_direction == "cross_to_right" and crossed_to_right) or (
                                                            alert_direction == "cross_to_left" and crossed_to_left))
                                if should_alert:
                                    behavior_logger.info(f"--- [方向性警報] --- 目標 ID: {track_id} 觸發了警戒線!")
                                    self.tripwire_alert_ids.add(track_id)
                                    self._set_event_type("tripwire_alert")
                                    break
                    self.track_last_positions[anchor_key] = current_anchor
                else:
                    continue
                break

    def _handle_dwell_logic(self, track_roi_status, current_time):
        if not Config.ROI_ENABLED: return
        current_tracked_ids = set(track_roi_status.keys())
        for track_id, is_in_roi in track_roi_status.items():
            if is_in_roi:
                if track_id not in self.dwell_time_trackers:
                    self.dwell_time_trackers[track_id] = {'start_time': current_time, 'alerted': False}
                else:
                    tracker_info = self.dwell_time_trackers[track_id]
                    if not tracker_info['alerted']:
                        dwell_duration = current_time - tracker_info['start_time']
                        if dwell_duration > Config.ROI_SETTINGS.get('dwell_time_threshold', 3.0):
                            behavior_logger.info(
                                f"--- [停留警報] --- 目標 ID: {track_id} 在 ROI 區域停留已超過 {Config.ROI_SETTINGS.get('dwell_time_threshold', 3.0)} 秒!")
                            self._set_event_type("dwell_alert")
            else:
                if track_id in self.dwell_time_trackers: del self.dwell_time_trackers[track_id]

        # 修正：移除未使用的變數
        disappeared_ids = set(self.dwell_time_trackers.keys()) - current_tracked_ids
        for track_id in disappeared_ids:
            del self.dwell_time_trackers[track_id]