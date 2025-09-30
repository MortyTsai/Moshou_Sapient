import json
import logging
import subprocess
import threading
import time
from queue import Queue
from typing import List, Dict, Any

import cv2
import numpy as np

from ..config import Config
from ..settings import settings


class ThreadedVideoCapture:
    """
    一個使用獨立執行緒預先讀取影片幀的輔助類別。
    透過將 I/O 操作與主處理邏輯分離，可以顯著減少等待時間，
    特別是在 CPU 密集型的影像處理任務中。
    """

    def __init__(self, source: str, max_queue_size: int = 256):
        """
        初始化 ThreadedVideoCapture。
        :param source: 影片檔案的路徑或串流 URL。
        :param max_queue_size: 內部佇列的最大尺寸。
        """
        self.cap = cv2.VideoCapture(source)
        self.q = Queue(maxsize=max_queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)

    def _update(self):
        """在背景執行緒中持續讀取幀並放入佇列。"""
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    self.q.put((False, None))  # 放入結束信號
                    return
                self.q.put((ret, frame))
            else:
                time.sleep(0.01)  # 佇列已滿，稍作等待

    def start(self):
        """啟動背景讀取執行緒。"""
        self.thread.start()
        return self

    def read(self):
        """從佇列中獲取一幀，此操作為阻塞式。"""
        return self.q.get()

    def is_opened(self) -> bool:
        """檢查影片擷取是否已成功開啟。"""
        return self.cap.isOpened()

    def release(self):
        """停止執行緒並釋放影片擷取資源。"""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()


def get_video_resolution(video_path: str) -> tuple[int, int] | None:
    """
    使用 ffprobe 高效地獲取影片的寬度和高度，無需解碼整個檔案。
    """
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        data = json.loads(result.stdout)
        if 'streams' in data and len(data['streams']) > 0:
            width = data['streams'][0].get('width')
            height = data['streams'][0].get('height')
            if width and height:
                return int(width), int(height)
        return None
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logging.error(f"[系統] 獲取影片解析度時出錯: {e}")
        return None


def draw_and_encode_segment(
        video_frames: Dict[int, np.ndarray],
        source_meta: Dict[str, Any],
        output_path: str,
        event_frames_data: List[Dict[str, Any]],
        event_type: str,
        event_start_frame_index: int
) -> bool:
    """從記憶體中的幀高效、同步地生成事件影片。"""
    if not event_frames_data or not video_frames:
        return False

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

        active_alert_ids = set()
        event_frames_indices = {f['frame_index'] for f in event_frames_data if
                                f['frame_index'] >= event_start_frame_index}

        for frame_data in event_frames_data:
            frame_idx = frame_data['frame_index']
            frame = video_frames.get(frame_idx)

            if frame is None:
                logging.warning(f"在記憶體中未找到幀 {frame_idx}，跳過此幀。")
                continue

            current_frame_index = frame_data['frame_index']
            if current_frame_index in event_frames_indices:
                for track in frame_data.get('tracks', []):
                    if track.get('has_crossed_tripwire'):
                        active_alert_ids.add(track['track_id'])

            overlay = frame.copy()

            if Config.ROI_ENABLED and Config.ROI_POLYGON_OBJECT:
                roi_points = np.array(Config.ROI_POLYGON_OBJECT.exterior.coords, dtype=np.int32)
                roi_points_scaled = (roi_points * np.array([scale_x, scale_y])).astype(np.int32)
                roi_overlay = overlay.copy()
                cv2.fillPoly(roi_overlay, [roi_points_scaled], color=(255, 255, 0))
                overlay = cv2.addWeighted(roi_overlay, 0.3, overlay, 0.7, 0)
                cv2.polylines(overlay, [roi_points_scaled], isClosed=True, color=(255, 255, 0), thickness=4)

            if Config.TRIPWIRES_ENABLED and Config.TRIPWIRE_LINE_OBJECTS:
                line_thickness, tip_length = 8, 0.02
                for tripwire_obj in Config.TRIPWIRE_LINE_OBJECTS:
                    line, direction = tripwire_obj['line'], tripwire_obj["direction"]
                    p1, p2 = np.array(line.coords[0]), np.array(line.coords[1])
                    p1_s = tuple((p1 * np.array([scale_x, scale_y])).astype(np.int32))
                    p2_s = tuple((p2 * np.array([scale_x, scale_y])).astype(np.int32))
                    if direction == "cross_to_right":
                        cv2.arrowedLine(overlay, p1_s, p2_s, (0, 0, 255), line_thickness, tipLength=tip_length)
                    elif direction == "cross_to_left":
                        cv2.arrowedLine(overlay, p2_s, p1_s, (0, 0, 255), line_thickness, tipLength=tip_length)
                    else:
                        cv2.line(overlay, p1_s, p2_s, (0, 0, 255), line_thickness)

            all_tracks = frame_data.get('tracks', [])
            active_tracks = [t for t in all_tracks if (t['track_id'] in active_alert_ids or
                                                       (t.get(
                                                           'is_in_roi') and current_frame_index in event_frames_indices))]
            inactive_tracks = [t for t in all_tracks if t not in active_tracks]

            temp_overlay = overlay.copy()
            for track in inactive_tracks:
                box = track['box_xyxy']
                x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])
                cv2.rectangle(temp_overlay, (x1, y1), (x2, y2), (128, 128, 128), 2)
            overlay = cv2.addWeighted(temp_overlay, 0.5, overlay, 0.5, 0)

            for track in active_tracks:
                track_id = track['track_id']
                box = track['box_xyxy']
                x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])
                box_color = (0, 0, 255) if track_id in active_alert_ids else (0, 255, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
                anchor_coords = track.get('anchors', [])
                for anchor in anchor_coords:
                    center_point = (int(anchor[0] * scale_x), int(anchor[1] * scale_y))
                    cv2.circle(overlay, center_point, 5, box_color, -1)
                cv2.putText(overlay, f'ID:{track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            info_panel = np.zeros((80, source_width, 3), dtype=np.uint8)
            frame[0:80, 0:source_width] = cv2.addWeighted(frame[0:80, 0:source_width], 0.5, info_panel, 0.5, 0)
            event_text = f"EVENT: {event_type.upper()}"
            cv2.putText(frame, event_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if current_frame_index >= event_start_frame_index:
                elapsed_time = (current_frame_index - event_start_frame_index) / source_fps
                duration_text = f"DURATION: {elapsed_time:.1f}s"
                cv2.putText(frame, duration_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if process.stdin:
                process.stdin.write(frame.tobytes())

        if process.stdin:
            process.stdin.close()

        stderr_output = process.communicate()[1]
        return_code = process.returncode

        if return_code != 0:
            logging.error(f"[FFmpeg] 編碼失敗 (返回碼: {return_code})")
            if stderr_output:
                logging.error(f"[FFmpeg] STDERR:\n{stderr_output.decode('utf-8', errors='ignore')}")
            return False

        logging.info(f"事件影片已成功儲存: {output_path}")
        return True

    except (BrokenPipeError, IOError) as e:
        logging.warning(f"[FFmpeg] 管道寫入時發生錯誤: {e}")
        return False
    finally:
        if process and process.poll() is None:
            process.kill()
            logging.warning("[FFmpeg] FFmpeg 程序被強制終止。")