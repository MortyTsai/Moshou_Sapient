import subprocess
import json
import logging
import time
import cv2
import threading
from queue import Queue
from typing import List, Dict, Any
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

    Args:
        video_path: 影片檔案的路徑。

    Returns:
        一個包含 (寬度, 高度) 的元組，如果失敗則返回 None。
    """
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=True, text=True)
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
    source_video_path: str,
    output_path: str,
    event_frames_data: List[Dict[str, Any]],
    event_type: str,
    event_start_frame_index: int
) -> bool:
    """
    高效處理事件影片片段的生成。

    此函式整合了多項性能優化策略：
    1.  **執行緒化預讀取**: 使用背景執行緒提前讀取影片幀，將 I/O 等待與 CPU 處理並行化。
    2.  **單次疊加層混合**: 將所有半透明的視覺元素先繪製到一個臨時疊加層上，
        然後一次性將其與主疊加層混合，極大地減少了昂貴的 Alpha Blending 操作次數。
    3.  **硬體加速編碼**: 利用 FFmpeg 和 NVIDIA 的 NVENC 進行高速影片編碼。

    Args:
        source_video_path: 原始影片檔案的路徑。
        output_path: 最終生成的事件影片的儲存路徑。
        event_frames_data: 包含每幀分析結果的列表。
        event_type: 事件的類型字串 (例如, "tripwire_alert")。
        event_start_frame_index: 事件正式開始的幀索引。

    Returns:
        如果影片成功生成則返回 True，否則返回 False。
    """
    if not event_frames_data:
        return False

    cap = ThreadedVideoCapture(source_video_path)
    if not cap.is_opened():
        logging.error(f"無法開啟影片檔案: {source_video_path}")
        return False
    cap.start()

    # 使用一個獨立的 VideoCapture 實例來安全地獲取元數據
    temp_cap = cv2.VideoCapture(source_video_path)
    if temp_cap.isOpened():
        source_fps = temp_cap.get(cv2.CAP_PROP_FPS) or 30.0
        source_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_cap.release()
    else:
        logging.warning("無法開啟臨時擷取器以讀取元數據，將使用預設值。")
        # 提供一個合理的預設值以防萬一
        source_fps, source_width, source_height = 30.0, 1920, 1080

    scale_x = source_width / settings.ANALYSIS_WIDTH
    scale_y = source_height / settings.ANALYSIS_HEIGHT
    output_fps = Config.TARGET_FPS if Config.VIDEO_FPS_MODE == "TARGET" and Config.TARGET_FPS > 0 else source_fps

    command = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{source_width}x{source_height}',
        '-pix_fmt', 'bgr24', '-r', str(source_fps),
        '-i', '-',
        '-c:v', 'hevc_nvenc', '-preset', 'p6'
    ]
    if Config.VIDEO_ENCODING_MODE == "BALANCED":
        bitrate_str = f"{Config.TARGET_BITRATE_MBPS}M"
        command.extend(['-rc', 'cbr', '-b:v', bitrate_str, '-maxrate', bitrate_str])
    else:
        quality_level = '23'
        command.extend(['-rc', 'vbr', '-cq', quality_level, '-b:v', '0', '-maxrate', '20M'])
    command.extend(['-r', str(output_fps), '-pix_fmt', 'yuv420p', output_path])

    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 手動跳過事件開始前的幀
    frames_to_skip = event_frames_data[0]['frame_index'] - 1
    for _ in range(frames_to_skip):
        ret, _ = cap.read()
        if not ret:
            logging.warning("在跳轉至事件起始幀時影片已結束。")
            break

    active_alert_ids = set()
    event_frames_indices = {f['frame_index'] for f in event_frames_data if f['frame_index'] >= event_start_frame_index}

    try:
        for frame_data in event_frames_data:
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("從執行緒佇列中讀取到結束信號或空幀，處理終止。")
                break

            current_frame_index = frame_data['frame_index']

            if current_frame_index in event_frames_indices:
                for track in frame_data.get('tracks', []):
                    if track.get('has_crossed_tripwire'):
                        active_alert_ids.add(track['track_id'])

            overlay = frame.copy()

            # 繪製靜態視覺元素 (ROI, Tripwires)
            if Config.ROI_ENABLED and Config.ROI_POLYGON_OBJECT:
                roi_points = np.array(Config.ROI_POLYGON_OBJECT.exterior.coords, dtype=np.int32)
                roi_points_scaled = (roi_points * np.array([scale_x, scale_y])).astype(np.int32)
                cv2.fillPoly(overlay, [roi_points_scaled], color=(255, 255, 0))
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

            # 分離活躍與非活躍物件以進行分層繪製
            all_tracks = frame_data.get('tracks', [])
            active_tracks = [t for t in all_tracks if (t['track_id'] in active_alert_ids or
                                                       (t.get('is_in_roi') and current_frame_index in event_frames_indices))]
            inactive_tracks = [t for t in all_tracks if t not in active_tracks]

            # 在臨時疊加層上繪製所有半透明的非活躍物件
            temp_overlay = overlay.copy()
            for track in inactive_tracks:
                box = track['box_xyxy']
                x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])
                cv2.rectangle(temp_overlay, (x1, y1), (x2, y2), (128, 128, 128), 2)

            # 執行一次混合操作來應用所有半透明效果
            overlay = cv2.addWeighted(temp_overlay, 0.5, overlay, 0.5, 0)

            # 在主疊加層上繪製不透明的活躍物件及其文字
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

            # 將包含所有視覺元素的疊加層與原始幀混合
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            # 繪製頂部狀態資訊面板
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
        return_code = process.wait()

        if return_code != 0:
            logging.error(f"[FFmpeg] 編碼失敗 (返回碼: {return_code})")
            return False

        logging.info(f"事件影片已成功儲存: {output_path}")
        return True

    except (BrokenPipeError, IOError) as e:
        logging.warning(f"[FFmpeg] 管道寫入時發生錯誤: {e}")
        return False
    finally:
        cap.release()
        if process.poll() is None:
            process.kill()