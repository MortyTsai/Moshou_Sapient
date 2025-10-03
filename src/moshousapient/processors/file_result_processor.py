# src/moshousapient/processors/file_result_processor.py
"""
此模組定義了 FileResultProcessor，負責處理從 isolated_inference_service
產生的 JSON 結果。它將結果分割成獨立的事件，並作為任務生產者，
將這些事件發送到任務佇列以供後續處理。
"""

# 1. 標準庫導入
import logging
import os
import pickle
import tempfile
from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# 3. 本專案相對導入
from ..config import Config
from ..services.task_queue_service import TaskQueueService


class FileResultProcessor:
    """
    負責處理從獨立推論服務產生的 JSON 結果的類別。
    它將連續的活躍幀序列分割成獨立的事件片段，並將每個片段
    作為一個任務發送到 TaskQueueService。
    """

    def __init__(self, notifier=None):
        """
        初始化 FileResultProcessor。

        :param notifier: 用於發送通知的通知器物件。
        """
        self.notifier = notifier
        self.task_queue = TaskQueueService()
        self.EVENT_TYPE_PRIORITY: Dict[str, int] = {
            "tripwire_alert": 2,
            "dwell_alert": 1,
        }
        logging.info("[FileResultProcessor] 已初始化。")

    @staticmethod
    def _is_frame_active(frame_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        判斷單一幀是否包含觸發事件的活動。

        :param frame_data: 單一幀的分析數據。
        :return: 一個元組 (是否活躍, 事件類型)。
        """
        active_event_type: Optional[str] = None
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
        將連續的活躍幀序列分割成獨立的事件片段，並處理最大時長分段。

        :param frames_data: 來自推論服務的完整幀數據列表。
        :param source_fps: 源影片的幀率。
        :return: 一個包含多個事件片段的列表。
        """
        if not frames_data or source_fps <= 0:
            return []

        events: List[Dict[str, Any]] = []
        in_event = False
        current_event_frames: List[Dict] = []

        pre_event_frames = int(Config.PRE_EVENT_SECONDS * source_fps)
        post_event_frames = int(Config.POST_EVENT_SECONDS * source_fps)
        max_event_frames = int(Config.MAX_EVENT_DURATION * source_fps)

        frame_buffer: deque = deque(maxlen=pre_event_frames)
        last_active_frame_idx = -1
        event_start_idx = -1

        for i, frame_data in enumerate(frames_data):
            is_active, _ = self._is_frame_active(frame_data)

            if is_active:
                last_active_frame_idx = i

            if not in_event and is_active:
                in_event = True
                event_start_idx = i
                current_event_frames.extend(frame_buffer)
                current_event_frames.append(frame_data)
            elif in_event:
                current_event_frames.append(frame_data)

            is_post_event_elapsed = (i > last_active_frame_idx) and (
                    (i - last_active_frame_idx) >= post_event_frames)
            is_max_duration_reached = in_event and (i - event_start_idx) >= max_event_frames

            if in_event and (is_post_event_elapsed or is_max_duration_reached):
                if current_event_frames:
                    highest_priority = -1
                    final_event_type = "person_detected"
                    event_start_frame = -1
                    for f in current_event_frames:
                        is_f_active, f_event_type = self._is_frame_active(f)
                        if is_f_active and f_event_type:
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

                if is_max_duration_reached:
                    in_event = True
                    overlap_frames_count = pre_event_frames
                    current_event_frames = list(deque(current_event_frames, maxlen=overlap_frames_count))
                    event_start_idx = i - len(current_event_frames) + 1
                else:
                    in_event = False
                    current_event_frames.clear()
                    frame_buffer.clear()

            if not in_event:
                frame_buffer.append(frame_data)

        if in_event and current_event_frames:
            highest_priority = -1
            final_event_type = "person_detected"
            event_start_frame = -1
            for f in current_event_frames:
                is_f_active, f_event_type = self._is_frame_active(f)
                if is_f_active and f_event_type:
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
        主處理函式，協調事件分割、並將事件作為任務分派到佇列。

        :param results: 來自 isolated_inference_service 的完整 JSON 結果。
        """
        source_video_path = results.get("video_path")
        analytics = results.get("analytics", {})
        frames_data = results.get("frames", [])

        if not frames_data or not source_video_path:
            logging.warning("[FileResultProcessor] 結果數據不完整，處理終止。")
            return

        source_fps = analytics.get('source_fps', 30.0)
        if source_fps <= 0:
            logging.warning(f"源影片幀率為零，將使用預設值 30.0 FPS。")
            source_fps = 30.0

        event_groups = self._segment_events(frames_data, source_fps=source_fps)

        if not event_groups:
            logging.warning("[FileResultProcessor] 分析完成，但未偵測到任何符合條件的事件片段。")
            return

        logging.info(f"[FileResultProcessor] 偵測到 {len(event_groups)} 個獨立事件，準備分派到任務佇列...")

        source_meta = {
            'width': analytics.get('source_width'),
            'height': analytics.get('source_height'),
            'fps': source_fps,
            'total_frames': analytics.get('total_frames')
        }

        for i, event_data in enumerate(event_groups):
            self._dispatch_video_task(event_data, source_video_path, source_meta, i + 1)

        logging.info("[FileResultProcessor] 所有事件已成功分派。")

    def _dispatch_video_task(self, event_data: Dict[str, Any], source_video_path: str,
                             source_meta: Dict[str, Any], event_index: int):
        """
        將單個事件組的元數據寫入臨時檔案，並將任務發送到任務佇列。
        """
        temp_file_path = ""
        try:
            event_type = event_data["event_type"]
            event_frames_metadata = [
                {k: v for k, v in frame.items() if k != 'frame'}
                for frame in event_data["frames"]
            ]

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', dir=str(Config.CAPTURES_DIR)) as temp_f:
                pickle.dump(event_frames_metadata, temp_f)
                temp_file_path = temp_f.name

            now = datetime.now()
            filename = f"{event_type}_{now.strftime('%Y%m%d_%H%M%S')}_evt{event_index}.mp4"
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
                "event_type": event_type,
                "source_meta": source_meta,
                "event_start_frame_index": event_data["event_start_frame"],
                "source_video_path": source_video_path,
                "rendering_config": rendering_config
            }

            payload_bytes = pickle.dumps(payload)

            task_id = self.task_queue.add_task(payload_bytes)
            if task_id:
                logging.info(f"已成功將事件 #{event_index} ('{event_type}') 作為任務 ID {task_id} 發送到佇列。")
            else:
                logging.error(f"將事件 #{event_index} 發送到任務佇列失敗，正在清理臨時檔案...")
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        except (IOError, pickle.PicklingError) as e:
            logging.error(f"建立事件任務 #{event_index} 時發生錯誤: {e}", exc_info=True)
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)