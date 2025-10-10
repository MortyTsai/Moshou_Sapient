# src/moshousapient/processors/file_event_producer.py
"""
定義了 FileEventProducer，負責處理來自獨立推論服務的 JSON 結果。
此生產者現在只負責識別連續的活動時段，並將整個時段的元數據
打包成單一任務，發送到任務佇列。事件分段的邏輯已移至消費者端。
"""

import logging
import pickle
import tempfile
import os
from typing import Dict, Any, List, Tuple, Optional

from ..configs.behavior_config import Config
from ..services.task_queue_service import TaskQueueService


class FileEventProducer:
    """
    負責處理從獨立推論服務產生的 JSON 結果的類別。
    """

    def __init__(self, notifier=None):
        """
        初始化 FileEventProducer。

        :param notifier: 用於發送通知的通知器物件。
        """
        self.notifier = notifier
        self.task_queue = TaskQueueService()
        self.EVENT_TYPE_PRIORITY: Dict[str, int] = {
            "tripwire_alert": 2,
            "dwell_alert": 1,
        }
        logging.debug("[FileEventProducer] 已初始化。")

    @staticmethod
    def _is_frame_active(frame_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        判斷單一幀是否包含觸發事件的活動。

        :param frame_data: 單一幀的分析數據。
        :return: 一個元組 (是否活躍, 事件類型)。
        """
        active_event_type: Optional[str] = None
        highest_priority = -1
        tracks = frame_data.get("tracks", [])
        if not tracks:
            return False, None

        for track in tracks:
            if track.get("has_crossed_tripwire"):
                if 2 > highest_priority:
                    highest_priority = 2
                    active_event_type = "tripwire_alert"
            if track.get("is_in_roi"):
                if 1 > highest_priority:
                    highest_priority = 1
                    active_event_type = "dwell_alert"

        return active_event_type is not None, active_event_type

    def _find_continuous_activity(self, frames_data: List[Dict], source_fps: float) -> List[Dict[str, Any]]:
        """
        從完整的幀數據中，找到所有連續的活動時段，並將每個時段作為一個事件返回。

        :param frames_data: 來自推論服務的完整幀數據列表。
        :param source_fps: 源影片的幀率。
        :return: 一個包含多個連續活動事件的列表。
        """
        if not frames_data or source_fps <= 0:
            return []

        events: List[Dict[str, Any]] = []
        in_activity = False
        current_activity_frames: List[Dict] = []

        pre_buffer_frames = int(Config.PRE_EVENT_SECONDS * source_fps)
        post_buffer_frames = int(Config.POST_EVENT_SECONDS * source_fps)

        frame_buffer = []
        post_activity_counter = 0

        for frame_data in frames_data:
            is_active, _ = self._is_frame_active(frame_data)

            if is_active:
                if not in_activity:
                    in_activity = True
                    current_activity_frames.extend(frame_buffer)
                    frame_buffer.clear()
                current_activity_frames.append(frame_data)
                post_activity_counter = 0
            else:
                if in_activity:
                    post_activity_counter += 1
                    current_activity_frames.append(frame_data)
                    if post_activity_counter >= post_buffer_frames:
                        events.append({"frames": list(current_activity_frames)})
                        current_activity_frames.clear()
                        in_activity = False
                else:
                    frame_buffer.append(frame_data)
                    if len(frame_buffer) > pre_buffer_frames:
                        frame_buffer.pop(0)

        if in_activity and current_activity_frames:
            events.append({"frames": list(current_activity_frames)})

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
            logging.warning("[FileEventProducer] 結果數據不完整，處理終止。")
            return

        source_fps = analytics.get("source_fps", 30.0)
        if source_fps <= 0:
            logging.warning("源影片幀率為零，將使用預設值 30.0 FPS。")
            source_fps = 30.0

        activity_groups = self._find_continuous_activity(frames_data, source_fps)

        if not activity_groups:
            logging.info("影片分析完成，未偵測到任何需要處理的活動。")
            return

        logging.info(f"偵測到 {len(activity_groups)} 個獨立活動時段，正在分派任務...")
        source_meta = {
            "width": analytics.get("source_width"),
            "height": analytics.get("source_height"),
            "fps": source_fps,
            "total_frames": analytics.get("total_frames"),
        }
        for i, activity_data in enumerate(activity_groups):
            self._dispatch_video_task(activity_data, source_video_path, source_meta, i + 1)

    def _dispatch_video_task(
        self,
        activity_data: Dict[str, Any],
        source_video_path: str,
        source_meta: Dict[str, Any],
        activity_index: int,
    ):
        """將單個活動時段的元數據寫入臨時檔案，並將任務發送到任務佇列。"""
        temp_file_path = ""
        try:
            event_frames_metadata = [
                {k: v for k, v in frame.items() if k != "frame"} for frame in activity_data["frames"]
            ]
            if not event_frames_metadata:
                return

            final_event_type = "person_detected"
            event_start_frame = -1
            highest_priority = -1
            for f in event_frames_metadata:
                is_f_active, f_event_type = self._is_frame_active(f)
                if is_f_active and f_event_type:
                    if event_start_frame == -1:
                        event_start_frame = f["frame_index"]
                    priority = self.EVENT_TYPE_PRIORITY.get(f_event_type, -1)
                    if priority > highest_priority:
                        highest_priority = priority
                        final_event_type = f_event_type

            if event_start_frame == -1:
                event_start_frame = event_frames_metadata[0].get("frame_index", 0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=str(Config.CAPTURES_DIR)) as temp_f:
                pickle.dump(event_frames_metadata, temp_f)
                temp_file_path = temp_f.name

            payload = {
                "data_path": temp_file_path,
                "event_type": final_event_type,
                "source_meta": source_meta,
                "event_start_frame_index": event_start_frame,
                "source_video_path": source_video_path,
            }
            payload_bytes = pickle.dumps(payload)
            task_id = self.task_queue.add_task(payload_bytes)

            if task_id:
                logging.debug(
                    f"已成功將活動 #{activity_index} ('{final_event_type}') 作為任務 ID {task_id} 發送到佇列。"
                )
            else:
                logging.error(f"將活動 #{activity_index} 發送到任務佇列失敗。")

        except (IOError, pickle.PicklingError) as e:
            logging.error(f"建立活動任務 #{activity_index} 時發生錯誤: {e}", exc_info=True)
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
