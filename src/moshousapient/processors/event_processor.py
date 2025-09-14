import logging
import time
from queue import Queue, Empty
from threading import Lock, Thread
from collections import deque

from ..config import Config
from .base_processor import BaseProcessor
from ..services.video_recorder import encode_and_send_video


class EventProcessor(BaseProcessor):
    def __init__(
            self,
            frame_queue: Queue,
            shared_state: dict,
            state_lock: Lock,
            notifier,
            active_recorders: list,
            video_fps_mode: str,
            target_fps: float,
            name: str = "EventProcessor"
    ):

        # print(f"DEBUG [event_processor.py]: Initialized with video_fps_mode = {video_fps_mode}")

        super().__init__(name)
        self.frame_queue = frame_queue
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.notifier = notifier
        self.active_recorders = active_recorders
        self.is_capturing_event = False
        self.last_person_seen_time = 0
        self.last_event_ended_time = 0
        self.event_start_time = 0
        buffer_size = int(Config.PRE_EVENT_SECONDS * Config.TARGET_FPS * 1.5)
        self.frame_buffer = deque(maxlen=buffer_size)
        self.event_recording = []
        self.current_event_features = []
        self.current_event_type = None
        self.dwell_time_trackers = {}
        self.track_last_positions = {}
        self.tripwire_alert_ids = set()
        self.video_fps_mode = video_fps_mode
        self.target_fps = target_fps

    def _target_func(self):
        logging.info(f"[{self.name}] 處理器已啟動。")

        while True:
            try:
                if self.stop_event.is_set() and self.frame_queue.empty():
                    break

                item = self.frame_queue.get(timeout=1)
                current_time = item['time']

                # --- 最終修正：在條件判斷前正確初始化所有變數 ---
                person_detected_now: bool = False
                current_tracks: list = []
                track_roi_status_now: dict = {}
                # --- 最終修正結束 ---

                if Config.VIDEO_SOURCE_TYPE == "FILE":
                    current_tracks = item.get('tracks', [])
                    person_detected_now = len(current_tracks) > 0
                    track_roi_status_now = item.get('track_roi_status', {})
                else:  # RTSP 模式
                    with self.state_lock:
                        current_tracks = self.shared_state.get('tracked_objects', [])
                        person_detected_now = self.shared_state.get('person_detected', False)
                        track_roi_status_now = self.shared_state.get('track_roi_status', {})

                self._handle_tripwire_logic(current_tracks)
                self._handle_dwell_logic(track_roi_status_now, current_time)

                frame_data = {
                    'frame': item['frame'], 'time': current_time,
                    'tracks': current_tracks,
                    'track_roi_status': track_roi_status_now,
                    'tripwire_alert_ids': self.tripwire_alert_ids.copy()
                }

                if self.is_capturing_event:
                    self.event_recording.append(frame_data)

                    reid_features_to_add = {}
                    if Config.VIDEO_SOURCE_TYPE == "FILE":
                        reid_features_to_add = item.get('reid_features_map', {})
                    else:  # RTSP 模式
                        with self.state_lock:
                            reid_features_to_add = self.shared_state.get('reid_features_map', {})

                    if reid_features_to_add:
                        self.current_event_features.extend(reid_features_to_add.values())
                else:
                    self.frame_buffer.append(frame_data)

                if person_detected_now:
                    self.last_person_seen_time = current_time

                self._update_event_state(person_detected_now, current_time)

            except Empty:
                if self.is_capturing_event:
                    self._update_event_state(False, time.time())
                continue
            except Exception as e:
                logging.error(f"[{self.name}] 執行緒發生未預期的錯誤: {e}", exc_info=True)
                time.sleep(1)

        if self.is_capturing_event:
            logging.info(f"[事件] 系統關閉，強制結束當前事件。")
            if len(self.event_recording) > 1:
                self._start_encoding_thread(list(self.event_recording))

        logging.info(f"[{self.name}] 處理器已停止。")

    def _handle_tripwire_logic(self, current_tracks):
        from shapely.geometry import Point, LineString
        from ..utils.geometry_utils import get_point_side_of_line
        if not Config.TRIPWIRES_ENABLED: return
        current_tracked_ids = {int(t[4]) for t in current_tracks}
        for track in current_tracks:
            x1, y1, x2, y2, track_id = track[:5]
            track_id = int(track_id)
            current_position = Point((x1 + x2) / 2, y2)
            last_position = self.track_last_positions.get(track_id)
            if last_position and last_position != current_position and Config.TRIPWIRE_LINE_OBJECTS:
                movement_line = LineString([last_position, current_position])
                for tripwire_obj in Config.TRIPWIRE_LINE_OBJECTS:
                    tripwire_line, alert_direction = tripwire_obj["line"], tripwire_obj["direction"]
                    if movement_line.intersects(tripwire_line):
                        p1, p2 = tripwire_line.coords
                        side_before = get_point_side_of_line(last_position, Point(p1), Point(p2))
                        side_after = get_point_side_of_line(current_position, Point(p1), Point(p2))
                        if side_before != 0 and side_after != 0 and side_before != side_after:
                            crossed_to_right = side_before == 1 and side_after == -1
                            crossed_to_left = side_before == -1 and side_after == 1
                            should_alert = (alert_direction == "both" or
                                            (alert_direction == "cross_to_right" and crossed_to_right) or
                                            (alert_direction == "cross_to_left" and crossed_to_left))
                            if should_alert:
                                logging.warning(f"--- [方向性警報] --- 目標 ID: {track_id} 觸發了警戒線!")
                                self.tripwire_alert_ids.add(track_id)
                                self._set_event_type("tripwire_alert")
                                break
            self.track_last_positions[track_id] = current_position
        disappeared_ids = set(self.track_last_positions.keys()) - current_tracked_ids
        for track_id in disappeared_ids:
            del self.track_last_positions[track_id]
            self.tripwire_alert_ids.discard(track_id)

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
                        if dwell_duration > Config.ROI_DWELL_TIME_THRESHOLD:
                            logging.warning(
                                f"--- [停留警報] --- 目標 ID: {track_id} 在 ROI 區域停留已超過 "
                                f"{Config.ROI_DWELL_TIME_THRESHOLD} 秒!")
                            tracker_info['alerted'] = True
                            self._set_event_type("dwell_alert")
            else:
                if track_id in self.dwell_time_trackers:
                    del self.dwell_time_trackers[track_id]
        disappeared_ids = set(self.dwell_time_trackers.keys()) - current_tracked_ids
        for track_id in disappeared_ids:
            del self.dwell_time_trackers[track_id]

    def _set_event_type(self, new_type: str):
        priority_map = {"tripwire_alert": 2, "dwell_alert": 1, "person_detected": 0}
        current_priority = priority_map.get(self.current_event_type, -1)
        new_priority = priority_map.get(new_type, -1)
        if new_priority > current_priority:
            if self.is_capturing_event:
                logging.info(f">>> [事件升級] '{self.current_event_type}' 事件已升級為 '{new_type}'")
            self.current_event_type = new_type

    def _update_event_state(self, person_detected_now, current_time):
        if not self.is_capturing_event:
            if person_detected_now and (current_time - self.last_event_ended_time > Config.COOLDOWN_PERIOD):
                self._set_event_type("person_detected")
                if self.current_event_type is not None:
                    logging.info(f">>> [事件] 偵測到 '{self.current_event_type}' 事件! 開始錄製...")
                    self.is_capturing_event = True
                    self.event_recording = list(self.frame_buffer)
                    self.event_start_time = self.event_recording[0]['time'] if self.event_recording else current_time
                    self.current_event_features.clear()
        else:
            should_end, end_reason = False, ""
            is_segmentation = False

            if not person_detected_now and (current_time - self.last_person_seen_time > Config.POST_EVENT_SECONDS):
                should_end, end_reason = True, "人物消失"
            elif current_time - self.event_start_time > Config.MAX_EVENT_DURATION:
                should_end, end_reason = True, "超過最大錄影時長"
                is_segmentation = True

            if should_end:
                logging.info(f"[事件] 事件結束 ({end_reason})。")
                completed_segment = list(self.event_recording)
                if len(completed_segment) > 1:
                    self._start_encoding_thread(completed_segment)

                if not is_segmentation:
                    self.is_capturing_event = False
                    self.event_recording.clear()
                    self.current_event_features.clear()
                    self.current_event_type = None
                    self.last_event_ended_time = current_time
                    with self.state_lock:
                        self.shared_state['event_ended'] = True
                else:
                    logging.info(">>> [事件] 進行事件分段，準備錄製下一段...")
                    buffer_frame_count = self.frame_buffer.maxlen
                    self.event_recording = completed_segment[-buffer_frame_count:]
                    self.event_start_time = self.event_recording[0]['time'] if self.event_recording else current_time
                    self.current_event_features.clear()

    def _start_encoding_thread(self, recording_segment: list):
        duration = recording_segment[-1]['time'] - recording_segment[0]['time']
        actual_fps = len(recording_segment) / duration if duration > 0 else self.target_fps
        features_copy = list(self.current_event_features)

        #print(f"DEBUG [event_processor.py]: Threading with video_fps_mode = {self.video_fps_mode}")

        thread_args = (
            recording_segment,  # -> frame_data_list
            self.notifier,  # -> notifier_instance
            actual_fps,  # -> actual_fps
            features_copy,  # -> reid_features_list
            self.current_event_type,  # -> event_type
            self.video_fps_mode,  # -> video_fps_mode
            self.target_fps  # -> target_fps
        )

        encoding_thread = Thread(
            target=encode_and_send_video,
            name="EncodingThread",
            args=thread_args,  # 使用我們定義好的元組
            daemon=True
        )
        self.active_recorders.append(encoding_thread)
        encoding_thread.start()