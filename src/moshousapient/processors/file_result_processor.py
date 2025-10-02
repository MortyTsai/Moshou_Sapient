# src/moshousapient/processors/file_result_processor.py
import logging
import os
import subprocess
from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

from ..config import Config
from ..services.database_service import process_reid_and_identify_person, save_event
from ..utils.video_utils import ThreadedVideoCapture
from ..settings import settings
from ..utils.visualization_utils import draw_static_overlays, draw_dynamic_overlays, draw_info_panel


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
        """判斷單一幀是否包含觸發事件的活動。"""
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
        """將連續的活躍幀序列分割成獨立的事件片段，並處理最大時長分段。"""
        if not frames_data or source_fps <= 0:
            return []

        events = []
        in_event = False
        current_event_frames = []

        pre_event_frames = int(Config.PRE_EVENT_SECONDS * source_fps)
        post_event_frames = int(Config.POST_EVENT_SECONDS * source_fps)
        max_event_frames = int(Config.MAX_EVENT_DURATION * source_fps)

        frame_buffer = deque(maxlen=pre_event_frames)
        last_active_frame_idx = -1
        event_start_idx = -1

        for i, frame_data in enumerate(frames_data):
            is_active, _ = self._is_frame_active(frame_data)

            if is_active:
                last_active_frame_idx = i

            # 狀態轉換：從非事件 -> 事件
            if not in_event and is_active:
                in_event = True
                event_start_idx = i
                current_event_frames.extend(frame_buffer)
                current_event_frames.append(frame_data)
            # 狀態維持：在事件中
            elif in_event:
                current_event_frames.append(frame_data)

                # 檢查是否需要結束或分段
                is_post_event_elapsed = (i > last_active_frame_idx) and (
                            (i - last_active_frame_idx) >= post_event_frames)
                is_max_duration_reached = (i - event_start_idx) >= max_event_frames

                if is_post_event_elapsed or is_max_duration_reached:
                    # 處理並儲存事件
                    if current_event_frames:
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

                    # 根據結束原因決定下一步
                    if is_max_duration_reached:  # 如果是因達到最大時長而分段
                        in_event = True  # 保持在事件狀態
                        overlap_frames_count = pre_event_frames
                        # 保留尾部的幀作為下一個分段的 pre_event buffer
                        current_event_frames = list(deque(current_event_frames, maxlen=overlap_frames_count))
                        event_start_idx = i - len(current_event_frames) + 1
                    else:  # 如果是因人物消失而結束
                        in_event = False
                        current_event_frames = []
                        frame_buffer.clear()

            # 狀態維持：不在事件中
            if not in_event:
                frame_buffer.append(frame_data)

        # 處理迴圈結束後可能還在進行中的最後一個事件
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
        """主處理函式，協調事件分割、影片預讀取和最終的繪製與編碼。"""
        source_video_path = results.get("video_path")
        analytics = results.get("analytics", {})
        frames_data = results.get("frames", [])

        if not frames_data or not source_video_path:
            return

        # 建立唯一的、可信的 source_fps 來源
        source_fps = analytics.get('source_fps', 30.0)
        if source_fps == 0:
            logging.warning("警告: 從分析結果中獲取的源影片幀率為 0，將使用預設值 30.0 FPS。")
            source_fps = 30.0

        event_groups = self._segment_events(frames_data, source_fps=source_fps)

        if not event_groups:
            logging.warning("分析完成，但未偵測到任何符合條件的事件片段。")
            return

        logging.info(f"正在將源影片 {os.path.basename(source_video_path)} 預讀取至記憶體...")
        cap = ThreadedVideoCapture(source_video_path)
        if not cap.is_opened():
            logging.error(f"無法在 FileResultProcessor 中開啟影片檔案: {source_video_path}")
            return

        # 不再依賴 cap.get(cv2.CAP_PROP_FPS)，強制使用來自 analytics 的 source_fps
        source_meta = {
            'fps': source_fps,
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

                success = self._draw_and_encode_segment(
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

    @staticmethod
    def _draw_and_encode_segment(
            video_frames: Dict[int, np.ndarray],
            source_meta: Dict[str, Any],
            output_path: str,
            event_frames_data: List[Dict[str, Any]],
            event_type: str,
            event_start_frame_index: int
    ) -> bool:
        """從記憶體中的幀高效、同步地生成事件影片，並使用統一的視覺化模組進行繪圖。"""
        process = None
        try:
            source_width = source_meta['width']
            source_height = source_meta['height']
            source_fps = source_meta['fps']

            scale_x = source_width / settings.ANALYSIS_WIDTH
            scale_y = source_height / settings.ANALYSIS_HEIGHT

            output_fps = Config.TARGET_FPS if Config.VIDEO_FPS_MODE == "TARGET" and Config.TARGET_FPS > 0 else source_fps

            command = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{source_width}x{source_height}',
                '-pix_fmt', 'bgr24', '-r', str(output_fps),
                '-i', '-',
                '-c:v', 'hevc_nvenc', '-preset', 'p6'
            ]
            if Config.VIDEO_ENCODING_MODE == "BALANCED":
                bitrate_str = f"{Config.TARGET_BITRATE_MBPS}M"
                command.extend(['-rc', 'cbr', '-b:v', bitrate_str, '-maxrate', bitrate_str])
            else:
                quality_level = '23'
                command.extend(['-rc', 'vbr', '-cq', quality_level, '-b:v', '0', '-maxrate', '20M'])
            command.extend(['-pix_fmt', 'yuv420p', output_path])

            process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            event_frames_indices = {f['frame_index'] for f in event_frames_data if
                                    f['frame_index'] >= event_start_frame_index}

            static_overlay_template = np.zeros((source_height, source_width, 3), dtype=np.uint8)
            static_overlay_template = draw_static_overlays(
                frame=static_overlay_template, scale_x=scale_x, scale_y=scale_y,
                roi_enabled=Config.ROI_ENABLED, roi_polygon=Config.ROI_POLYGON_OBJECT,
                tripwires_enabled=Config.TRIPWIRES_ENABLED, tripwire_line_objects=Config.TRIPWIRE_LINE_OBJECTS,
                tripwire_line_thickness=settings.TRIPWIRE_LINE_THICKNESS,
                tripwire_tip_length=settings.TRIPWIRE_TIP_LENGTH
            )

            frame_ratio = source_fps / output_fps if Config.VIDEO_FPS_MODE == "TARGET" else 1.0
            frame_cursor = 0.0

            for frame_data in event_frames_data:
                if Config.VIDEO_FPS_MODE == "TARGET":
                    frame_cursor += 1.0
                    if frame_cursor < frame_ratio:
                        continue
                    frame_cursor -= frame_ratio

                frame_idx = frame_data['frame_index']
                frame = video_frames.get(frame_idx)
                if frame is None:
                    continue

                overlay = cv2.add(frame, static_overlay_template)
                current_frame_index = frame_data['frame_index']
                active_alert_ids, active_roi_ids = set(), set()
                if current_frame_index in event_frames_indices:
                    for track in frame_data.get('tracks', []):
                        if track.get('has_crossed_tripwire'):
                            active_alert_ids.add(track['track_id'])
                        if track.get('is_in_roi'):
                            active_roi_ids.add(track['track_id'])

                all_tracks = frame_data.get('tracks', [])
                overlay = draw_dynamic_overlays(overlay, all_tracks, active_alert_ids, active_roi_ids, scale_x, scale_y)
                final_frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

                elapsed_time = -1.0
                if current_frame_index >= event_start_frame_index:
                    elapsed_frames = current_frame_index - event_start_frame_index
                    elapsed_time = elapsed_frames / source_fps

                final_frame = draw_info_panel(final_frame, event_type, elapsed_time)

                if process.stdin:
                    process.stdin.write(final_frame.tobytes())

            if process.stdin:
                process.stdin.close()
            stderr_output = process.communicate()[1]
            return_code = process.returncode

            if return_code != 0:
                logging.error(f"[FFmpeg] 編碼失敗 (返回碼: {return_code})")
                if stderr_output:
                    logging.error(f"[FFmpeg] STDERR:\n{stderr_output.decode('utf-8', errors='ignore')}")
                return False
            return True
        except (BrokenPipeError, IOError) as e:
            logging.warning(f"[FFmpeg] 管道寫入時發生錯誤: {e}")
            return False
        finally:
            if process and process.poll() is None:
                process.kill()
                logging.warning("[FFmpeg] FFmpeg 程序被強制終止。")