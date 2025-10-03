# src/moshousapient/workers/video_processing_worker.py
"""
此模組定義了 VideoProcessingWorker，一個獨立的背景工作程序。
它負責從任務佇列中獲取影片處理任務，並執行高負載的繪圖與編碼工作。
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
from typing import Dict, List

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
from moshousapient.settings import settings
from moshousapient.utils.video_utils import ThreadedVideoCapture
from moshousapient.utils.visualization_utils import (
    draw_static_overlays,
    draw_dynamic_overlays,
    draw_info_panel
)


class VideoProcessingWorker:
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
        """
        優雅地處理關閉信號。

        :param signum: 信號編號。
        :param _frame: 當前的堆疊幀 (未使用)。
        """
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

    def _process_video_task(self, task_payload_bytes: bytes) -> bool:
        """
        處理單個影片任務的核心邏輯（繪圖與編碼）。

        :param task_payload_bytes: 從佇列中獲取的原始二進位 payload。
        :return: 處理成功返回 True，否則返回 False。
        """
        try:
            payload = pickle.loads(task_payload_bytes)
        except pickle.UnpicklingError as e:
            logging.error(f"[Worker-{self.worker_id}] 反序列化任務 payload 失敗: {e}", exc_info=True)
            return True

        data_path = payload.get("data_path")
        if not data_path:
            logging.error(f"[Worker-{self.worker_id}] 任務 payload 缺少 'data_path'。")
            return True

        event_frames_data: List[Dict] = []
        try:
            output_path = payload.get("output_path")
            event_type = payload.get("event_type", "unknown_event")
            event_start_frame_index = payload.get("event_start_frame_index", -1)
            source_meta = payload.get("source_meta", {})
            source_video_path = payload.get("source_video_path")
            rendering_config = payload.get("rendering_config", {})

            if not output_path:
                logging.error(f"[Worker-{self.worker_id}] 任務 payload 缺少 'output_path'。")
                return False

            try:
                with open(data_path, 'rb') as f:
                    event_frames_data = pickle.load(f)
            except (IOError, pickle.UnpicklingError) as e:
                logging.error(f"[Worker-{self.worker_id}] 無法讀取或反序列化幀數據檔案: {e}", exc_info=True)
                return False

            if not event_frames_data:
                logging.warning(f"[Worker-{self.worker_id}] 幀數據為空，不進行處理。")
                return True

            if source_video_path:
                event_frames_data = self._load_file_mode_frames(event_frames_data, source_video_path)
                if not event_frames_data:
                    return False

            source_width = source_meta.get('width') or event_frames_data[0]['frame'].shape[1]
            source_height = source_meta.get('height') or event_frames_data[0]['frame'].shape[0]
            source_fps = source_meta.get('fps') or 30.0

            scale_x = source_width / settings.ANALYSIS_WIDTH
            scale_y = source_height / settings.ANALYSIS_HEIGHT

            output_fps = settings.TARGET_FPS if settings.VIDEO_FPS_MODE == "TARGET" and settings.TARGET_FPS > 0 else source_fps

            ffmpeg_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{source_width}x{source_height}',
                '-pix_fmt', 'bgr24', '-r', str(output_fps), '-i', '-',
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
                    f"[Worker-{self.worker_id}] 開始為事件 '{event_type}' 繪製並編碼 {len(event_frames_data)} 幀...")
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

                frame_ratio = source_fps / output_fps if settings.VIDEO_FPS_MODE == "TARGET" else 1.0
                frame_cursor = 0.0

                for frame_data in event_frames_data:
                    if settings.VIDEO_FPS_MODE == "TARGET":
                        frame_cursor += 1.0
                        if frame_cursor < frame_ratio:
                            continue
                        frame_cursor -= frame_ratio

                    frame = frame_data['frame']
                    overlay = cv2.add(frame, static_overlay)

                    all_tracks = frame_data.get('tracks', [])

                    active_alert_ids = {
                        track['track_id'] for track in all_tracks if track.get('has_crossed_tripwire')
                    }
                    active_roi_ids = {
                        track['track_id'] for track in all_tracks if track.get('is_in_roi')
                    }

                    overlay = draw_dynamic_overlays(overlay, all_tracks, active_alert_ids, active_roi_ids, scale_x,
                                                    scale_y)
                    final_frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

                    elapsed_time = -1.0
                    current_frame_index = frame_data.get('frame_index')
                    if event_start_frame_index != -1 and current_frame_index is not None:
                        elapsed_frames = current_frame_index - event_start_frame_index
                        elapsed_time = elapsed_frames / source_fps if source_fps > 0 else 0.0
                    elif 'time' in frame_data:
                        event_start_time = event_frames_data[0].get('time', frame_data['time'])
                        elapsed_time = frame_data['time'] - event_start_time

                    final_frame = draw_info_panel(final_frame, event_type, elapsed_time)

                    if process.stdin:
                        process.stdin.write(final_frame.tobytes())

                if process.stdin:
                    process.stdin.close()

                stderr_output = process.communicate()[1]
                if process.returncode != 0:
                    logging.error(f"[Worker-{self.worker_id}] FFmpeg 編碼失敗 (返回碼: {process.returncode})")
                    if stderr_output:
                        logging.error(f"[FFmpeg STDERR]:\n{stderr_output.decode('utf-8', errors='ignore')}")
                    return False

                logging.info(f"[Worker-{self.worker_id}] 事件影片已成功儲存至: {output_path}")
                return True

            except (IOError, BrokenPipeError) as e:
                logging.warning(f"[Worker-{self.worker_id}] FFmpeg 管道寫入時發生錯誤: {e}")
                return False
            finally:
                if process and process.poll() is None:
                    logging.warning(f"[Worker-{self.worker_id}] 強制終止 FFmpeg 程序。")
                    process.kill()

        finally:
            if data_path and os.path.exists(data_path):
                try:
                    os.remove(data_path)
                    logging.debug(f"[Worker-{self.worker_id}] 已清理臨時數據檔案: {data_path}")
                except OSError as e:
                    logging.warning(f"[Worker-{self.worker_id}] 清理臨時檔案 {data_path} 失敗: {e}")

    def run(self):
        """ Worker 的主執行迴圈。 """
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
                        success = self._process_video_task(task['payload'])
                        if success:
                            self.task_queue.complete_task(task['id'])
                        else:
                            logging.warning(f"[Worker-{self.worker_id}] 任務 ID: {task['id']} 處理失敗。")
                            self.task_queue.fail_task(task['id'])
                    else:
                        time.sleep(2)

                except Exception as e:
                    logging.error(f"[Worker-{self.worker_id}] 在處理任務時發生未預期的錯誤: {e}", exc_info=True)
                    if task and 'id' in task:
                        logging.error(f"[Worker-{self.worker_id}] 將對異常任務 ID: {task['id']} 進行失敗重排。")
                        self.task_queue.fail_task(task['id'])
                    time.sleep(5)
        finally:
            self.task_queue.close_connection()
            logging.info(f"[Worker-{self.worker_id}] 主迴圈已停止，程序即將退出。")


def main():
    """
    Worker 程序的入口點。
    """
    from moshousapient.logging_setup import setup_logging
    setup_logging()

    worker_id = f"{os.getpid()}"
    worker = VideoProcessingWorker(worker_id=worker_id)
    worker.run()


if __name__ == "__main__":
    main()