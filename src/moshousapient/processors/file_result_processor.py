# src/moshousapient/processors/file_result_processor.py

import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np

from ..config import Config
from ..utils.video_utils import draw_and_encode_segment
from ..services.database_service import process_reid_and_identify_person, save_event


class FileResultProcessor:
    def __init__(self, notifier=None):
        self.notifier = notifier
        self.EVENT_TYPE_PRIORITY = {
            "tripwire_alert": 2,
            "dwell_alert": 1,
        }
        logging.info("[FileResultProcessor] 已初始化。")

    @staticmethod
    def _is_frame_active(frame_data: Dict[str, Any]) -> Tuple[bool, str | None]:
        """
        判斷一個幀是否活躍，並返回活躍原因（事件類型）。
        活躍定義：任何追蹤目標觸發了 ROI 或警戒線。
        """
        active_event_type = None
        highest_priority = -1

        if not frame_data or not frame_data.get('tracks'):
            return False, None

        for track in frame_data['tracks']:
            if track.get('has_crossed_tripwire'):
                if 2 > highest_priority:
                    highest_priority = 2
                    active_event_type = "tripwire_alert"
            if track.get('is_in_roi'):
                if 1 > highest_priority:
                    highest_priority = 1
                    active_event_type = "dwell_alert"

        return active_event_type is not None, active_event_type

    def _segment_events(self, frames_data: List[Dict], source_fps: float) -> List[Dict[str, Any]]:
        if not frames_data:
            return []

        trigger_points = []
        frame_activity = [self._is_frame_active(f) for f in frames_data]

        is_currently_active = False
        for i, (is_active, event_type) in enumerate(frame_activity):
            if is_active and not is_currently_active:
                trigger_points.append({
                    "frame_index": frames_data[i]['frame_index'],
                    "event_type": event_type
                })
            is_currently_active = is_active

        if not trigger_points:
            logging.info("未偵測到任何有效的事件觸發點。")
            return []

        pre_frames = int(Config.PRE_EVENT_SECONDS * source_fps)
        post_frames = int(Config.POST_EVENT_SECONDS * source_fps)

        intervals = []
        for point in trigger_points:
            start_idx = point["frame_index"] - 1
            end_idx = start_idx
            for i in range(start_idx, len(frame_activity)):
                if frame_activity[i][0]:
                    end_idx = i
                elif (i - end_idx) > post_frames:
                    break

            start_frame = frames_data[start_idx]['frame_index']
            end_frame = frames_data[end_idx]['frame_index']

            intervals.append([
                max(1, start_frame - pre_frames),
                min(len(frames_data), end_frame + post_frames)
            ])

        if not intervals:
            return []

        intervals.sort(key=lambda x: x[0])
        merged_intervals = [intervals[0]]
        for current in intervals[1:]:
            last = merged_intervals[-1]
            if current[0] <= last[1]:
                last[1] = max(last[1], current[1])
            else:
                merged_intervals.append(current)

        events = []
        frames_map = {f['frame_index']: f for f in frames_data}
        activity_map = {frames_data[i]['frame_index']: activity for i, activity in enumerate(frame_activity)}

        for start, end in merged_intervals:
            event_frames = [frames_map[i] for i in range(start, end + 1) if i in frames_map]

            highest_priority = -1
            final_event_type = "unknown_event"
            for i in range(start, end + 1):
                if i in activity_map:
                    is_active, event_type = activity_map[i]
                    if is_active:
                        priority = self.EVENT_TYPE_PRIORITY.get(event_type, -1)
                        if priority > highest_priority:
                            highest_priority = priority
                            final_event_type = event_type

            if event_frames:
                events.append({
                    "frames": event_frames,
                    "event_type": final_event_type
                })

        logging.info(f"事件分段完成，共偵測到 {len(events)} 個獨立事件。")
        return events

    def process_results(self, results: Dict[str, Any]):
        source_video_path = results.get("video_path")
        analytics = results.get("analytics", {})
        frames_data = results.get("frames", [])
        source_fps = analytics.get('source_fps', 30.0)

        if not frames_data:
            return

        event_groups = self._segment_events(frames_data, source_fps)

        for i, event_data in enumerate(event_groups):
            event_frames = event_data["frames"]
            event_type = event_data["event_type"]

            logging.info(f"正在處理事件 #{i + 1}/{len(event_groups)} (類型: {event_type})...")

            now = datetime.now()
            filename = f"{event_type}_{now.strftime('%Y%m%d_%H%M%S')}_evt{i + 1}.mp4"
            output_path = os.path.join(Config.CAPTURES_DIR, filename)

            success = draw_and_encode_segment(
                source_video_path=source_video_path,
                output_path=output_path,
                event_frames_data=event_frames,
                all_frames_data=frames_data,
                output_fps=int(Config.TARGET_FPS),
                pre_event_sec=Config.PRE_EVENT_SECONDS,
                post_event_sec=Config.POST_EVENT_SECONDS
            )

            if success:
                all_features = [np.array(track['feature']) for frame in event_frames for track in frame['tracks'] if
                                track.get('feature')]
                person_id = None
                if all_features:
                    person_id = process_reid_and_identify_person(all_features)
                save_event(output_path, event_type, person_id)
                if self.notifier:
                    message = f"**事件警報!**\n類型: `{event_type}`\n來源: `{os.path.basename(source_video_path)}`"
                    self.notifier.schedule_notification(message, file_path=output_path)
            else:
                logging.error(f"事件 #{i + 1} 的影片片段生成失敗。")

        logging.info("所有事件已處理完畢。")