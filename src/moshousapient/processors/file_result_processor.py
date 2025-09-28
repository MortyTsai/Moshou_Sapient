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
        if not frames_data or source_fps <= 0: return []
        frame_activity = [self._is_frame_active(f) for f in frames_data]
        active_indices = [i for i, (is_active, _) in enumerate(frame_activity) if is_active]
        if not active_indices: return []

        max_frames_total = int(Config.MAX_EVENT_DURATION * source_fps)
        pre_frames = int(Config.PRE_EVENT_SECONDS * source_fps)
        post_frames = int(Config.POST_EVENT_SECONDS * source_fps)
        max_frames_core = max(1, max_frames_total - pre_frames - post_frames)

        events = []
        ptr = 0
        while ptr < len(active_indices):
            core_start_index = active_indices[ptr]
            core_end_index = min(core_start_index + max_frames_core - 1, len(frames_data) - 1)

            actual_core_end_index = core_start_index
            next_ptr = ptr
            for i in range(ptr, len(active_indices)):
                if active_indices[i] <= core_end_index:
                    actual_core_end_index = active_indices[i]
                    next_ptr = i
                else:
                    break

            read_start_index = max(0, core_start_index - pre_frames)
            read_end_index = min(len(frames_data), actual_core_end_index + post_frames + 1)

            segment_frames = frames_data[read_start_index:read_end_index]

            highest_priority = -1
            final_event_type = "person_detected"
            for i in range(core_start_index, actual_core_end_index + 1):
                _, event_type = frame_activity[i]
                priority = self.EVENT_TYPE_PRIORITY.get(event_type, -1)
                if priority > highest_priority:
                    highest_priority = priority
                    final_event_type = event_type

            if segment_frames:
                events.append({
                    "frames": segment_frames,
                    "event_type": final_event_type,
                    "event_start_frame": frames_data[core_start_index]['frame_index']
                })
            ptr = next_ptr + 1
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
            now = datetime.now()
            event_type = event_data["event_type"]
            filename = f"{event_type}_{now.strftime('%Y%m%d_%H%M%S')}_evt{i + 1}.mp4"
            output_path = os.path.join(Config.CAPTURES_DIR, filename)

            logging.info(f"正在處理事件 #{i + 1}/{len(event_groups)} (類型: {event_type})...")
            success = draw_and_encode_segment(
                source_video_path=source_video_path,
                output_path=output_path,
                event_frames_data=event_data["frames"],
                event_type=event_data["event_type"],
                event_start_frame_index=event_data["event_start_frame"]
            )

            if success:
                all_features = [np.array(track['feature']) for frame in event_data["frames"] for track in
                                frame['tracks'] if
                                track.get('feature')]
                person_id = None
                if all_features:
                    person_id = process_reid_and_identify_person(all_features)
                save_event(output_path, event_type, person_id)
                if self.notifier:
                    message = f"**事件警報!**\n 類型: `{event_type}` \n 來源: `{os.path.basename(source_video_path)}`"
                    self.notifier.schedule_notification(message, file_path=output_path)
            else:
                logging.error(f"事件 #{i + 1} 的影片片段生成失敗。")

        logging.info("所有事件已處理完畢。")