# src/moshousapient/processors/file_result_processor.py
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

from ..config import Config
from ..utils.video_utils import draw_and_encode_segment
from ..services.database_service import process_reid_and_identify_person, save_event


class FileResultProcessor:
    def __init__(self, notifier=None):
        self.notifier = notifier
        logging.info("[FileResultProcessor] 已初始化。")

    @staticmethod
    def _segment_events(frames_data: List[Dict], source_fps: float) -> List[List[Dict]]:
        if not frames_data: return []

        events, current_event_frames = [], []
        last_active_frame = -1

        active_frames_set = {f['frame_index'] for f in frames_data if f['tracks']}
        if not active_frames_set: return []

        total_frames = frames_data[-1]['frame_index']

        for frame_index in range(1, total_frames + 1):
            is_active = frame_index in active_frames_set

            if is_active:
                if not current_event_frames:
                    if last_active_frame == -1 or (frame_index - last_active_frame) > (
                            Config.COOLDOWN_PERIOD * source_fps):
                        current_event_frames.append(frames_data[frame_index - 1])
                else:
                    current_event_frames.append(frames_data[frame_index - 1])

            if current_event_frames:
                if is_active:
                    last_active_frame = frame_index
                else:
                    if (frame_index - last_active_frame) > (Config.POST_EVENT_SECONDS * source_fps):
                        events.append(current_event_frames)
                        current_event_frames = []

        if current_event_frames:
            events.append(current_event_frames)

        logging.info(f"事件分段完成，共偵測到 {len(events)} 個獨立事件。")
        return events

    def process_results(self, results: Dict[str, Any]):
        source_video_path = results.get("video_path")
        analytics = results.get("analytics", {})
        frames_data = results.get("frames", [])
        source_fps = analytics.get('source_fps', 30.0)

        if not frames_data: return

        event_groups = self._segment_events(frames_data, source_fps)

        for i, event_frames in enumerate(event_groups):
            logging.info(f"正在處理事件 #{i + 1}/{len(event_groups)}...")
            now = datetime.now()
            filename = f"event_{now.strftime('%Y%m%d_%H%M%S')}_evt{i + 1}.mp4"
            output_path = os.path.join(Config.CAPTURES_DIR, filename)

            success = draw_and_encode_segment(
                source_video_path=source_video_path,
                output_path=output_path,
                event_frames_data=event_frames,
                output_fps=int(Config.TARGET_FPS),
                pre_event_sec=Config.PRE_EVENT_SECONDS,
                post_event_sec=Config.POST_EVENT_SECONDS
            )

            if success:
                all_features = [np.array(track['feature']) for frame in event_frames for track in frame['tracks'] if
                                track.get('feature')]
                person_id = process_reid_and_identify_person(all_features)
                save_event(output_path, "person_detected_file_mode", person_id)
                if self.notifier:
                    message = f"**事件警報!**\n來源: `{os.path.basename(source_video_path)}`"
                    self.notifier.schedule_notification(message, file_path=output_path)
            else:
                logging.error(f"事件 #{i + 1} 的影片片段生成失敗。")
        logging.info("所有事件已處理完畢。")