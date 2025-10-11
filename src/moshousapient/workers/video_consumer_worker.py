# src/moshousapient/workers/video_consumer_worker.py
"""
一個獨立的背景工作程序 (消費者)，負責處理影片編碼任務。

此 Worker 是事件處理的統一中心，負責從佇列獲取任務，並統一執行
事件分段、視覺化配置生成、幀抽樣、繪圖與編碼。
"""

# 1. 標準庫導入
import logging
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 2. 第三方庫導入
import cv2
import numpy as np

# 確保在作為獨立腳本執行時能找到 moshousapient 模組
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 3. 本專案導入
from moshousapient.configs.behavior_config import Config
from moshousapient.configs.logging_config import configure_logging_for_queue
from moshousapient.configs.settings_config import settings
from moshousapient.services.database_models import Event
from moshousapient.services.database_service import SessionLocal
from moshousapient.services.notification_service import NotificationService
from moshousapient.services.task_queue_service import TaskQueueService
from moshousapient.utils.video_io_utils import ThreadedVideoCapture
from moshousapient.utils.visualization_utils import (
    draw_dynamic_overlays,
    draw_info_panel,
    draw_static_overlays,
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
        self.hostname = os.uname().nodename if hasattr(os, "uname") else "windows"
        self.notifier: Optional[NotificationService] = None
        if Config.DISCORD_ENABLED:
            if Config.DISCORD_TOKEN and Config.DISCORD_CHANNEL_ID:
                self.notifier = NotificationService(token=Config.DISCORD_TOKEN, channel_id=Config.DISCORD_CHANNEL_ID)
            else:
                logging.warning(f"[Worker-{self.worker_id}] Discord 功能已啟用，但未提供完整的憑證。通知功能將被禁用。")
        logging.debug(f"[Worker-{self.worker_id}] 初始化完成。")

    def _handle_shutdown_signal(self, signum, _frame):
        """優雅地處理關閉信號。"""
        logging.info(f"[Worker-{self.worker_id}] 收到關閉信號 {signum}，將在完成當前任務後退出。")
        self.stop_event.set()

    def _load_file_mode_frames(self, frames_metadata: List[Dict], video_path: str) -> List[Dict]:
        """為 FILE 模式的任務，從源影片中讀取實際的幀圖像數據。"""
        logging.debug(f"[Worker-{self.worker_id}] 正在為 FILE 模式預讀取源影片: {os.path.basename(video_path)}")
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
        logging.debug(f"[Worker-{self.worker_id}] 影片預讀取完成，共 {len(video_frames)} 幀。")

        hydrated_frames_data = []
        for frame_meta in frames_metadata:
            frame_index = frame_meta.get("frame_index")
            if frame_index in video_frames:
                frame_meta["frame"] = video_frames[frame_index]
                hydrated_frames_data.append(frame_meta)
        return hydrated_frames_data

    @staticmethod
    def _segment_event_frames(all_frames: List[Dict], source_fps: float) -> List[List[Dict]]:
        """將一個長的事件幀列表，切分成多個符合 MAX_EVENT_DURATION 的子事件列表。"""
        max_frames = int(settings.MAX_EVENT_DURATION * source_fps)
        overlap_frames = int(settings.PRE_EVENT_SECONDS * source_fps)

        if not all_frames or len(all_frames) <= max_frames:
            return [all_frames]

        logging.debug(f"長事件偵測到 ({len(all_frames)} 幀)，將其切分為多個 {settings.MAX_EVENT_DURATION} 秒的片段。")
        segments = []
        start_index = 0
        while start_index < len(all_frames):
            end_index = start_index + max_frames
            segment = all_frames[start_index:end_index]
            segments.append(segment)
            if end_index >= len(all_frames):
                break
            start_index += max_frames - overlap_frames
        return segments

    def _load_and_prepare_task_data(self, payload: Dict[str, Any]) -> Tuple[Optional[List[Dict]], Dict, str]:
        """從 payload 載入並準備幀數據、元數據和事件類型。"""
        data_path = payload.get("data_path")
        if not data_path:
            raise ValueError("Missing 'data_path' in payload")

        with open(data_path, "rb") as f:
            event_frames_metadata = pickle.load(f)

        if not event_frames_metadata:
            logging.warning(f"[Worker-{self.worker_id}] 幀數據為空，任務將被跳過。")
            return None, {}, ""

        source_video_path = payload.get("source_video_path")
        event_frames_data = (
            self._load_file_mode_frames(event_frames_metadata, source_video_path)
            if source_video_path
            else event_frames_metadata
        )

        source_meta = payload.get("source_meta", {})
        event_type = payload.get("event_type", "unknown_event")
        return event_frames_data, source_meta, event_type

    def _process_video_task(self, task_payload_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """處理單個影片任務的核心邏輯（分段、繪圖與編碼）。"""
        data_path: Optional[str] = None
        try:
            payload = pickle.loads(task_payload_bytes)
            data_path = payload.get("data_path")

            event_frames_data, source_meta, event_type = self._load_and_prepare_task_data(payload)
            if not event_frames_data:
                return True, None

            source_fps = source_meta.get("fps") or 30.0
            event_segments = self._segment_event_frames(event_frames_data, source_fps)

            for i, segment_frames in enumerate(event_segments):
                if "time" in segment_frames[0]:
                    now = datetime.fromtimestamp(segment_frames[0]["time"])
                else:
                    now = datetime.now()

                segment_suffix = f"_seg{i + 1}" if len(event_segments) > 1 else ""
                filename = f"{event_type}_{now.strftime('%Y%m%d_%H%M%S')}{segment_suffix}.mp4"
                output_path = os.path.join(settings.CAPTURES_DIR, filename)

                success, error_msg = self._encode_segment(segment_frames, output_path, event_type, source_meta)
                if not success:
                    return False, f"Segment #{i + 1} encoding failed: {error_msg}"

        except (pickle.UnpicklingError, IOError, ValueError) as e:
            return False, f"Failed to read/prepare task data: {e}"
        except Exception:
            logging.exception(f"[Worker-{self.worker_id}] Critical error in _process_video_task")
            return False, "Unhandled exception"
        else:
            return True, None
        finally:
            if data_path and os.path.exists(data_path):
                try:
                    os.remove(data_path)
                except OSError as e:
                    logging.warning(f"[Worker-{self.worker_id}] Failed to clean up temp file {data_path}: {e}")

    @staticmethod
    def _build_ffmpeg_command(output_path: str, source_width: int, source_height: int, source_fps: float) -> List[str]:
        """根據設定構建 FFmpeg 編碼命令。"""
        is_target_mode = settings.VIDEO_FPS_MODE == "TARGET" and settings.TARGET_FPS > 0
        output_fps = settings.TARGET_FPS if is_target_mode else source_fps

        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{source_width}x{source_height}",
            "-pix_fmt",
            "bgr24",
            "-framerate",
            str(source_fps),
            "-i",
            "-",
        ]

        if is_target_mode and abs(source_fps - output_fps) > 0.1:
            command.extend(["-vf", f"fps={output_fps}"])
        else:
            command.extend(["-r", str(output_fps)])

        command.extend(["-c:v", "hevc_nvenc", "-preset", "p6", "-an"])

        if settings.VIDEO_ENCODING_MODE == "BALANCED":
            bitrate_str = f"{settings.TARGET_BITRATE_MBPS}M"
            command.extend(["-rc", "cbr", "-b:v", bitrate_str, "-maxrate", bitrate_str])
        else:
            quality_level = "23"
            command.extend(["-rc", "vbr", "-cq", quality_level, "-b:v", "0", "-maxrate", "20M"])

        command.extend(["-pix_fmt", "yuv420p", output_path])
        return command

    def _render_frame_overlays(self, frame: np.ndarray, frame_data: Dict, scale_x: float, scale_y: float) -> np.ndarray:
        """在單一幀上繪製所有靜態和動態疊加層。"""
        static_overlay = np.zeros_like(frame)
        static_overlay = draw_static_overlays(
            frame=static_overlay,
            scale_x=scale_x,
            scale_y=scale_y,
            roi_enabled=Config.ROI_ENABLED,
            roi_polygon=Config.ROI_POLYGON_OBJECT,
            tripwires_enabled=Config.TRIPWIRES_ENABLED,
            tripwire_line_objects=Config.TRIPWIRE_LINE_OBJECTS,
            tripwire_line_thickness=settings.TRIPWIRE_LINE_THICKNESS,
            tripwire_tip_length=settings.TRIPWIRE_TIP_LENGTH,
        )
        overlay = cv2.add(frame, static_overlay)

        all_tracks = frame_data.get("tracks", [])
        active_alert_ids = {t["track_id"] for t in all_tracks if t.get("has_crossed_tripwire")}
        active_roi_ids = {t["track_id"] for t in all_tracks if t.get("is_in_roi")}
        overlay = draw_dynamic_overlays(overlay, all_tracks, active_alert_ids, active_roi_ids, scale_x, scale_y)
        return overlay

    def _finalize_segment_processing(self, event_type: str, output_path: str):
        """在編碼成功後，記錄日誌、發送通知並寫入資料庫。"""
        logging.info(f"[Worker-{self.worker_id}] 事件影片已儲存至: {os.path.basename(output_path)}")
        self._notify_and_log_event(event_type, output_path)

    def _run_ffmpeg_process(
        self,
        ffmpeg_cmd: List[str],
        segment_frames: List[Dict],
        source_fps: float,
        scale_x: float,
        scale_y: float,
        event_type: str,
    ) -> Tuple[bool, Optional[str]]:
        """執行 FFmpeg 子程序並處理 I/O。"""
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        for i, frame_data in enumerate(segment_frames):
            rendered_frame = self._render_frame_overlays(frame_data["frame"], frame_data, scale_x, scale_y)
            final_frame = draw_info_panel(rendered_frame, event_type, i / source_fps)
            if process.stdin:
                process.stdin.write(final_frame.tobytes())

        if process.stdin:
            process.stdin.close()

        stderr_output = process.communicate(timeout=60)[1]
        if process.returncode != 0:
            error_msg = f"FFmpeg encoding failed (code: {process.returncode})"
            if stderr_output:
                logging.error(f"[FFmpeg STDERR]:\n{stderr_output.decode('utf-8', errors='ignore')}")
            return False, error_msg

        return True, None

    def _encode_segment(
        self, segment_frames: List[Dict], output_path: str, event_type: str, source_meta: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """對單個事件分段進行繪圖和編碼。"""
        if not segment_frames:
            return True, None

        source_width = source_meta.get("width") or segment_frames[0]["frame"].shape[1]
        source_height = source_meta.get("height") or segment_frames[0]["frame"].shape[0]
        source_fps = source_meta.get("fps") or 30.0
        scale_x = source_width / settings.ANALYSIS_WIDTH
        scale_y = source_height / settings.ANALYSIS_HEIGHT

        ffmpeg_cmd = self._build_ffmpeg_command(output_path, source_width, source_height, source_fps)

        try:
            logging.debug(f"[Worker-{self.worker_id}] 開始為事件 '{event_type}' 的分段編碼 {len(segment_frames)} 幀...")
            success, error_msg = self._run_ffmpeg_process(
                ffmpeg_cmd, segment_frames, source_fps, scale_x, scale_y, event_type
            )
            if not success:
                return False, error_msg

            self._finalize_segment_processing(event_type, output_path)

        except (IOError, BrokenPipeError) as e:
            return False, f"FFmpeg pipe error: {e}"
        except subprocess.TimeoutExpired:
            logging.exception(f"FFmpeg process timed out for {output_path}")
            return False, "FFmpeg process timed out"
        else:
            return True, None

    def _notify_and_log_event(self, event_type: str, output_path: str):
        """發送通知並將事件記錄到資料庫。"""
        if self.notifier:
            message = (
                f"**MoshouSapient 安全警報**\n"
                f"> **事件類型**: `{event_type}`\n"
                f"> **發生時間**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
                f"> **處理單元**: `Worker-{self.worker_id}@{self.hostname}`"
            )
            self.notifier.schedule_notification(message=message, file_path=output_path)

        db = None
        try:
            db = SessionLocal()
            new_event = Event(event_type=event_type, video_path=output_path)
            db.add(new_event)
            db.commit()
            logging.info(f"[Worker-{self.worker_id}] 事件記錄已寫入資料庫。")
        except Exception:
            logging.exception(f"[Worker-{self.worker_id}] 寫入資料庫時發生錯誤")
            if db:
                db.rollback()
        finally:
            if db:
                db.close()

    def _handle_single_task(self, task: Dict[str, Any]):
        """處理單個從佇列中獲取的任務。"""
        logging.debug(f"[Worker-{self.worker_id}] 已獲取任務 ID: {task['id']}")

        try:
            payload = pickle.loads(task["payload"])
            if payload.get("task_type") == "file_inference":
                log_message = (
                    f"[Worker-{self.worker_id}] Task ID: {task['id']} is 'file_inference' type, "
                    f"releasing for Scheduler."
                )
                logging.debug(log_message)
                self.task_queue.fail_task(task["id"], "Requeued for Scheduler", requeue=True)
                time.sleep(3)
                return
        except Exception:
            pass

        success, error_message = self._process_video_task(task["payload"])
        if success:
            self.task_queue.complete_task(task["id"])
        else:
            final_error_msg = error_message or "Worker processing failed"
            logging.warning(f"[Worker-{self.worker_id}] 任務 ID: {task['id']} 處理失敗。")
            self.task_queue.fail_task(task["id"], final_error_msg)

    def run(self):
        """Worker 的主執行迴圈。"""
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

        logging.debug(f"[Worker-{self.worker_id}] 在主機 {self.hostname} 上開始監聽任務...")
        if self.notifier:
            self.notifier.start()
            time.sleep(5)

        try:
            while not self.stop_event.is_set():
                task = None
                try:
                    task = self.task_queue.reserve_task(self.worker_id)
                    if task:
                        self._handle_single_task(task)
                    else:
                        time.sleep(2)
                except Exception:
                    logging.exception(f"[Worker-{self.worker_id}] Unhandled error in main loop")
                    if task and "id" in task:
                        self.task_queue.fail_task(task["id"], "Unhandled exception")
                    time.sleep(5)
        finally:
            if self.notifier:
                logging.debug(f"[Worker-{self.worker_id}] 正在停止通知服務...")
                self.notifier.stop()
            self.task_queue.close_connection()
            logging.debug(f"[Worker-{self.worker_id}] 主迴圈已停止，程序即將退出。")


def worker_entrypoint(log_queue):
    """
    multiprocessing.Process 的目標函數，用於啟動單個 Worker。
    """
    configure_logging_for_queue(log_queue)
    Config.initialize_static_settings()
    worker_id = f"{os.getpid()}"
    worker = VideoConsumerWorker(worker_id=worker_id)
    worker.run()
