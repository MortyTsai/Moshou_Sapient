import logging
import os
from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

from ..config import Config
from ..services.database_service import process_reid_and_identify_person, save_event
from ..utils.video_utils import ThreadedVideoCapture, draw_and_encode_segment


class FileResultProcessor:
    """負責處理從獨立推論服務產生的 JSON 結果的類別。"""

    def __init__(self, notifier=None):
        """初始化 FileResultProcessor。"""
        self.notifier = notifier
        self.EVENT_TYPE_PRIORITY = {
            "tripwire_alert": 2,
            "dwell_alert": 1,
        }
        logging.info("[FileResultProcessor] 已初始化。")

    @staticmethod
    def _is_frame_active(frame_data: Dict[str, Any]) -> Tuple[bool, str | None]:
        """
        判斷單一幀是否包含觸發事件的活動。
        此函式會解析從子程序傳來的、包含詳細字典的軌跡列表。
        """
        active_event_type = None
        highest_priority = -1

        tracks = frame_data.get('tracks', [])
        if not tracks:
            return False, None

        for track in tracks:
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
        """
        將連續的活躍幀序列分割成獨立的事件片段。
        此函式實現了狀態變化檢測，以避免對同一個持續事件重複創建分段。
        """
        if not frames_data or source_fps <= 0:
            return []

        events = []
        in_event = False
        current_event_frames = []

        pre_frames = int(Config.PRE_EVENT_SECONDS * source_fps)
        post_frames = int(Config.POST_EVENT_SECONDS * source_fps)

        frame_buffer = deque(maxlen=pre_frames)

        last_active_frame_idx = -1

        for i, frame_data in enumerate(frames_data):
            is_active, _ = self._is_frame_active(frame_data)

            if is_active:
                last_active_frame_idx = i

            if not in_event and is_active:
                in_event = True
                current_event_frames.extend(frame_buffer)
                current_event_frames.append(frame_data)
            elif in_event:
                current_event_frames.append(frame_data)

                if i > last_active_frame_idx and (i - last_active_frame_idx) >= post_frames:
                    in_event = False

                    highest_priority = -1
                    final_event_type = "person_detected"
                    event_start_frame = -1

                    for f in current_event_frames:
                        is_f_active, f_event_type = self._is_frame_active(f)
                        if is_f_active:
                            if event_start_frame == -1:
                                event_start_frame = f['frame_index']
                            priority = self.EVENT_TYPE_PRIORITY.get(f_event_type, -1)
                            if priority > highest_priority:
                                highest_priority = priority
                                final_event_type = f_event_type

                    if event_start_frame != -1:
                        events.append({
                            "frames": list(current_event_frames),
                            "event_type": final_event_type,
                            "event_start_frame": event_start_frame
                        })

                    current_event_frames = []
                    frame_buffer.clear()

            if not in_event:
                frame_buffer.append(frame_data)

        if in_event and current_event_frames:
            highest_priority = -1
            final_event_type = "person_detected"
            event_start_frame = -1
            for f in current_event_frames:
                is_f_active, f_event_type = self._is_frame_active(f)
                if is_f_active:
                    if event_start_frame == -1:
                        event_start_frame = f['frame_index']
                    priority = self.EVENT_TYPE_PRIORITY.get(f_event_type, -1)
                    if priority > highest_priority:
                        highest_priority = priority
                        final_event_type = f_event_type

            if event_start_frame != -1:
                events.append({
                    "frames": list(current_event_frames),
                    "event_type": final_event_type,
                    "event_start_frame": event_start_frame
                })

        return events

    def process_results(self, results: Dict[str, Any]):
        """
        主處理函式，協調事件分割、影片預讀取和最終的繪製與編碼。
        """
        source_video_path = results.get("video_path")
        analytics = results.get("analytics", {})
        frames_data = results.get("frames", [])

        if not frames_data:
            return

        event_groups = self._segment_events(frames_data, source_fps=analytics.get('source_fps', 30.0))

        if not event_groups:
            logging.warning("分析完成，但未偵測到任何符合條件的事件片段。")
            return

        logging.info(f"正在將源影片 {os.path.basename(source_video_path)} 預讀取至記憶體...")
        cap = ThreadedVideoCapture(source_video_path)
        if not cap.is_opened():
            logging.error(f"無法在 FileResultProcessor 中開啟影片檔案: {source_video_path}")
            return

        source_meta = {
            'fps': cap.cap.get(cv2.CAP_PROP_FPS) or 30.0,
            'width': int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

        cap.start()

        video_frames = {}
        for i in range(source_meta['total_frames']):
            ret, frame = cap.read()
            if not ret:
                break
            video_frames[i + 1] = frame
        cap.release()
        logging.info(f"影片預讀取完成，共 {len(video_frames)} 幀。")

        try:
            for i, event_data in enumerate(event_groups):
                now = datetime.now()
                event_type = event_data["event_type"]
                filename = f"{event_type}_{now.strftime('%Y%m%d_%H%M%S')}_evt{i + 1}.mp4"
                output_path = os.path.join(Config.CAPTURES_DIR, filename)

                logging.info(f"正在處理事件 #{i + 1}/{len(event_groups)} (類型: {event_type})...")

                success = draw_and_encode_segment(
                    video_frames=video_frames,
                    source_meta=source_meta,
                    output_path=output_path,
                    event_frames_data=event_data["frames"],
                    event_type=event_data["event_type"],
                    event_start_frame_index=event_data["event_start_frame"]
                )

                if success:
                    all_features = [np.array(track['feature']) for frame in event_data["frames"] for track in
                                    frame['tracks'] if track.get('feature')]
                    person_id = None
                    if all_features:
                        person_id = process_reid_and_identify_person(all_features)

                    save_event(output_path, event_type, person_id)
                    if self.notifier:
                        message = (f"**事件警報!**\n"
                                   f"類型: `{event_type}`\n"
                                   f"來源: `{os.path.basename(source_video_path)}`")
                        self.notifier.schedule_notification(message, file_path=output_path)
                else:
                    logging.error(f"事件 #{i + 1} 的影片片段生成失敗。")
        finally:
            logging.info("所有事件已處理完畢。")