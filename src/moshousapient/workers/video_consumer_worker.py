# src/moshousapient/workers/video_consumer_worker.py
"""
定義了 VideoConsumerWorker，一個獨立的背景工作程序 (消費者)。

此 Worker 現在是事件處理的統一中心。它負責：
1. 從任務佇列獲取包含事件元數據的任務。
2. 根據需要從源影片加載實際幀數據 (FILE 模式)。
3. 將可能很長的事件幀列表，根據 MAX_EVENT_DURATION 切分成標準時長的「子事件」。
4. 為每個子事件生成影片檔名和視覺化配置。
5. 對每個子事件的幀進行繪製和編碼，生成最終的 .mp4 影片。
"""

# 1. 標準庫導入
import logging
import signal
import time
import pickle
import os
import sys
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 2. 第三方庫導入
import cv2
import numpy as np

# 為了確保在作為獨立腳本執行時能找到 moshousapient 模組
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 3. 本專案相對導入
from moshousapient.services.task_queue_service import TaskQueueService
from moshousapient.configs.settings_config import settings
from moshousapient.configs.behavior_config import Config
from moshousapient.utils.video_io_utils import ThreadedVideoCapture
from moshousapient.utils.visualization_utils import (
    draw_static_overlays,
    draw_dynamic_overlays,
    draw_info_panel
)


class VideoConsumerWorker:
    """
    一個影片處理工作程序，負責消費來自任務佇列的事件。
    """

    def __init__(self, worker_id: str):
        """
        初始化 Worker。

        :param worker_id: 此 Worker 的唯一識別碼。
        """
        self.worker_id = worker_id
        self.task_queue = TaskQueueService()
        self.stop_event = threading.Event()
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'windows'
        logging.info(f"[Worker-{self.worker_id}] 初始化完成。")

    def _handle_shutdown_signal(self, signum, _frame):
        """優雅地處理關閉信號。"""
        logging.info(f"[Worker-{self.worker_id}] 收到關閉信號 {signum}，將在完成當前任務後退出。")
        self.stop_event.set()

    def _load_file_mode_frames(self, frames_metadata: List[Dict], video_path: str) -> List[Dict]:
        """
        為 FILE 模式的任務，從源影片中讀取實際的幀圖像數據。

        :param frames_metadata: 只包含元數據的幀列表。
        :param video_path: 源影片的路徑。
        :return: 包含了實際 'frame' 圖像數據的完整幀列表。
        """
        logging.info(f"[Worker-{self.worker_id}] 正在為 FILE 模式預讀取源影片: {os.path.basename(video_path)}")
        cap = ThreadedVideoCapture(video_path)
        if not cap.is_opened():
            logging.error(f"[Worker-{self.worker_id}] 無法開啟影片檔案: {video_path}")
            return []

        video_frames = {}
        total_frames = int(cap.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.start()
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            video_frames[i + 1] = frame
        cap.release()
        logging.info(f"[Worker-{self.worker_id}] 影片預讀取完成，共 {len(video_frames)} 幀。")

        hydrated_frames_data = []
        for frame_meta in frames_metadata:
            frame_index = frame_meta.get('frame_index')
            if frame_index in video_frames:
                frame_meta['frame'] = video_frames[frame_index]
                hydrated_frames_data.append(frame_meta)
        return hydrated_frames_data

    @staticmethod
    def _segment_event_frames(all_frames: List[Dict], source_fps: float) -> List[List[Dict]]:
        """
        將一個長的事件幀列表，切分成多個符合 MAX_EVENT_DURATION 的子事件列表。

        :param all_frames: 包含單個完整活動的所有幀的列表。
        :param source_fps: 原始影片的幀率。
        :return: 一個列表，其中每個元素都是一個子事件的幀列表。
        """
        max_frames = int(settings.MAX_EVENT_DURATION * source_fps)
        overlap_frames = int(settings.PRE_EVENT_SECONDS * source_fps)

        if len(all_frames) <= max_frames:
            return [all_frames]

        logging.info(f"長事件偵測到 ({len(all_frames)} 幀)，將其切分為多個 {settings.MAX_EVENT_DURATION} 秒的片段。")

        segments = []
        start_index = 0
        while start_index < len(all_frames):
            end_index = start_index + max_frames
            segment = all_frames[start_index:end_index]
            segments.append(segment)

            if end_index >= len(all_frames):
                break

            start_index += max_frames - overlap_frames
            if start_index >= len(all_frames):
                break

        return segments

    def _process_video_task(self, task_payload_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """
        處理單個影片任務的核心邏輯（分段、繪圖與編碼）。

        :param task_payload_bytes: 從佇列中獲取的原始二進位 payload。
        :return: 一個元組 (is_success, error_message)。
        """
        data_path: Optional[str] = None
        try:
            # --- 1. Payload 解析與驗證 ---
            try:
                payload = pickle.loads(task_payload_bytes)
            except pickle.UnpicklingError as e:
                return False, f"反序列化任務 payload 失敗: {e}"

            data_path = payload.get("data_path")
            if not data_path:
                return False, "任務 payload 缺少 'data_path'。"

            event_type = payload.get("event_type", "unknown_event")
            source_meta = payload.get("source_meta", {})
            source_video_path = payload.get("source_video_path")

            # --- 2. 幀數據加載 ---
            try:
                with open(data_path, 'rb') as f:
                    event_frames_metadata = pickle.load(f)
            except (IOError, pickle.UnpicklingError) as e:
                return False, f"無法讀取或反序列化幀數據檔案: {e}"

            if not event_frames_metadata:
                logging.warning(f"[Worker-{self.worker_id}] 幀數據為空，任務視為成功完成。")
                return True, None

            if source_video_path:
                event_frames_data = self._load_file_mode_frames(event_frames_metadata, source_video_path)
            else:
                event_frames_data = event_frames_metadata

            if not event_frames_data:
                return False, "加載幀數據後列表為空，處理終止。"

            # --- 3. 統一事件分段 ---
            source_fps = source_meta.get('fps') or 30.0
            event_segments = self._segment_event_frames(event_frames_data, source_fps)

            # --- 消費者端統一生成 rendering_config ---
            rendering_config = {
                "roi_enabled": Config.ROI_ENABLED,
                "roi_polygon": Config.ROI_POLYGON_OBJECT,
                "tripwires_enabled": Config.TRIPWIRES_ENABLED,
                "tripwire_line_objects": Config.TRIPWIRE_LINE_OBJECTS,
            }

            # --- 4. 遍歷分段並編碼 ---
            for i, segment_frames in enumerate(event_segments):
                now = datetime.fromtimestamp(segment_frames[0]['time']) if 'time' in segment_frames[
                    0] else datetime.now()
                segment_suffix = f"_seg{i + 1}" if len(event_segments) > 1 else ""
                filename = f"{event_type}_{now.strftime('%Y%m%d_%H%M%S')}{segment_suffix}.mp4"
                output_path = os.path.join(settings.CAPTURES_DIR, filename)

                success, error_msg = self._encode_segment(
                    segment_frames, output_path, event_type, source_meta, rendering_config
                )
                if not success:
                    return False, f"分段 #{i + 1} 編碼失敗: {error_msg}"

            return True, None

        except Exception as e:
            logging.error(f"[Worker-{self.worker_id}] 在 _process_video_task 中發生嚴重錯誤: {e}", exc_info=True)
            return False, f"未處理的異常: {e}"

        finally:
            # --- 5. 清理 ---
            if data_path and os.path.exists(data_path):
                try:
                    os.remove(data_path)
                    logging.debug(f"[Worker-{self.worker_id}] 已清理臨時數據檔案: {data_path}")
                except OSError as e:
                    logging.warning(f"[Worker-{self.worker_id}] 清理臨時檔案 {data_path} 失敗: {e}")

    def _encode_segment(self, segment_frames: List[Dict], output_path: str, event_type: str,
                        source_meta: Dict, rendering_config: Dict) -> Tuple[bool, Optional[str]]:
        """
        對單個事件分段進行繪製和編碼。

        :param segment_frames: 包含單個事件分段所有幀的列表。
        :param output_path: 最終影片的輸出路徑。
        :param event_type: 事件類型。
        :param source_meta: 原始影片的元數據。
        :param rendering_config: 繪圖所需的配置。
        :return: 一個元組 (is_success, error_message)。
        """
        if not segment_frames:
            return True, None

        source_width = source_meta.get('width') or segment_frames[0]['frame'].shape[1]
        source_height = source_meta.get('height') or segment_frames[0]['frame'].shape[0]
        source_fps = source_meta.get('fps') or 30.0
        scale_x = source_width / settings.ANALYSIS_WIDTH
        scale_y = source_height / settings.ANALYSIS_HEIGHT
        output_fps = settings.TARGET_FPS if settings.VIDEO_FPS_MODE == "TARGET" and settings.TARGET_FPS > 0 else source_fps

        if settings.VIDEO_FPS_MODE == "TARGET":
            target_duration = len(segment_frames) / source_fps
            target_total_frames = int(target_duration * output_fps)

            if len(segment_frames) > target_total_frames > 0:
                indices = np.linspace(0, len(segment_frames) - 1, target_total_frames, dtype=int)
                segment_frames = [segment_frames[i] for i in indices]

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{source_width}x{source_height}',
            '-pix_fmt', 'bgr24',
            '-framerate', str(output_fps),
            '-i', '-',
            '-r', str(output_fps),
            '-c:v', 'hevc_nvenc', '-preset', 'p6', '-an'
        ]

        if settings.VIDEO_ENCODING_MODE == "BALANCED":
            bitrate_str = f"{settings.TARGET_BITRATE_MBPS}M"
            ffmpeg_cmd.extend(['-rc', 'cbr', '-b:v', bitrate_str, '-maxrate', bitrate_str])
        else:
            quality_level = '23'
            ffmpeg_cmd.extend(['-rc', 'vbr', '-cq', quality_level, '-b:v', '0', '-maxrate', '20M'])
        ffmpeg_cmd.extend(['-pix_fmt', 'yuv420p', output_path])

        process = None
        try:
            logging.info(
                f"[Worker-{self.worker_id}] 開始為事件 '{event_type}' 的分段編碼 "
                f"{len(segment_frames)} 幀 (輸出 FPS: {output_fps})..."
            )
            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            static_overlay = np.zeros((source_height, source_width, 3), dtype=np.uint8)
            static_overlay = draw_static_overlays(
                frame=static_overlay, scale_x=scale_x, scale_y=scale_y,
                roi_enabled=rendering_config.get("roi_enabled", False),
                roi_polygon=rendering_config.get("roi_polygon"),
                tripwires_enabled=rendering_config.get("tripwires_enabled", False),
                tripwire_line_objects=rendering_config.get("tripwire_line_objects", []),
                tripwire_line_thickness=settings.TRIPWIRE_LINE_THICKNESS,
                tripwire_tip_length=settings.TRIPWIRE_TIP_LENGTH
            )

            for i, frame_data in enumerate(segment_frames):
                frame = frame_data['frame']
                overlay = cv2.add(frame, static_overlay)
                all_tracks = frame_data.get('tracks', [])
                active_alert_ids = {t['track_id'] for t in all_tracks if t.get('has_crossed_tripwire')}
                active_roi_ids = {t['track_id'] for t in all_tracks if t.get('is_in_roi')}
                overlay = draw_dynamic_overlays(overlay, all_tracks, active_alert_ids, active_roi_ids, scale_x, scale_y)

                elapsed_time = i / output_fps
                final_frame = draw_info_panel(overlay, event_type, elapsed_time)

                if process.stdin:
                    process.stdin.write(final_frame.tobytes())

            if process.stdin:
                process.stdin.close()

            stderr_output = process.communicate()[1]
            if process.returncode != 0:
                error_msg = f"FFmpeg 編碼失敗 (返回碼: {process.returncode})"
                if stderr_output:
                    logging.error(f"[FFmpeg STDERR]:\n{stderr_output.decode('utf-8', errors='ignore')}")
                return False, error_msg

            logging.info(f"[Worker-{self.worker_id}] 事件影片分段已成功儲存至: {output_path}")
            return True, None

        except (IOError, BrokenPipeError) as e:
            return False, f"FFmpeg 管道寫入時發生錯誤: {e}"
        finally:
            if process and process.poll() is None:
                logging.warning(f"[Worker-{self.worker_id}] 強制終止 FFmpeg 程序。")
                process.kill()

    def run(self):
        """ Worker 的主執行迴圈。"""
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        logging.info(f"[Worker-{self.worker_id}] 在主機 {self.hostname} 上開始監聽任務...")

        try:
            while not self.stop_event.is_set():
                task = None
                try:
                    task = self.task_queue.reserve_task(self.worker_id)
                    if task:
                        logging.info(f"[Worker-{self.worker_id}] 已獲取任務 ID: {task['id']}")
                        success, error_message = self._process_video_task(task['payload'])

                        if success:
                            self.task_queue.complete_task(task['id'])
                        else:
                            final_error_msg = error_message or "Worker processing failed without specific error"
                            logging.warning(f"[Worker-{self.worker_id}] 任務 ID: {task['id']} 處理失敗。")
                            self.task_queue.fail_task(task['id'], final_error_msg)
                    else:
                        time.sleep(2)
                except Exception as e:
                    logging.error(f"[Worker-{self.worker_id}] 在處理任務時發生未預期的錯誤: {e}", exc_info=True)
                    if task and 'id' in task:
                        error_message = f"Unhandled exception in worker loop: {e}"
                        logging.error(f"[Worker-{self.worker_id}] 將對異常任務 ID: {task['id']} 進行失敗重排。")
                        self.task_queue.fail_task(task['id'], error_message)
                    time.sleep(5)
        finally:
            self.task_queue.close_connection()
            logging.info(f"[Worker-{self.worker_id}] 主迴圈已停止，程序即將退出。")


def main():
    """ Worker 程序的入口點。"""
    from moshousapient.configs.logging_config import setup_logging
    setup_logging()

    # [修復] 確保子程序在啟動時加載行為設定
    from moshousapient.configs.behavior_config import Config
    Config.initialize_static_settings()

    worker_id = f"{os.getpid()}"
    worker = VideoConsumerWorker(worker_id=worker_id)
    worker.run()


if __name__ == "__main__":
    main()