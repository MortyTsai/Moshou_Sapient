import subprocess
import json
import logging
import cv2
from typing import List, Dict, Any
import numpy as np

from ..settings import settings
from ..config import Config


def get_video_resolution(video_path: str) -> tuple[int, int] | None:
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
        source_video_path: str,
        output_path: str,
        event_frames_data: List[Dict[str, Any]],
        event_type: str,  # <--- 恢復這個參數
        event_start_frame_index: int
) -> bool:
    if not event_frames_data: return False
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened(): return False

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cap.set(cv2.CAP_PROP_POS_FRAMES, event_frames_data[0]['frame_index'] - 1)

    active_alert_ids = set()
    event_frames_indices = {f['frame_index'] for f in event_frames_data if f['frame_index'] >= event_start_frame_index}

    try:
        for frame_data in event_frames_data:
            ret, frame = cap.read()
            if not ret: break

            current_frame_index = frame_data['frame_index']

            if current_frame_index in event_frames_indices:
                for track in frame_data.get('tracks', []):
                    if track.get('has_crossed_tripwire'):
                        active_alert_ids.add(track['track_id'])

            overlay = frame.copy()

            if Config.ROI_ENABLED and Config.ROI_POLYGON_OBJECT:
                roi_points = np.array(Config.ROI_POLYGON_OBJECT.exterior.coords, dtype=np.int32)
                roi_points_scaled = (roi_points * np.array([scale_x, scale_y])).astype(np.int32)
                cv2.fillPoly(overlay, [roi_points_scaled], color=(255, 255, 0))
                cv2.polylines(overlay, [roi_points_scaled], isClosed=True, color=(255, 255, 0), thickness=4)

            if Config.TRIPWIRES_ENABLED and Config.TRIPWIRE_LINE_OBJECTS:
                line_thickness, tip_length = 8, 0.02
                for tripwire_obj in Config.TRIPWIRE_LINE_OBJECTS:
                    line, direction = tripwire_obj["line"], tripwire_obj["direction"]
                    p1, p2 = np.array(line.coords[0]), np.array(line.coords[1])
                    p1_s = tuple((p1 * np.array([scale_x, scale_y])).astype(np.int32))
                    p2_s = tuple((p2 * np.array([scale_x, scale_y])).astype(np.int32))
                    if direction == "cross_to_right":
                        cv2.arrowedLine(overlay, p1_s, p2_s, (0, 0, 255), line_thickness, tipLength=tip_length)
                    elif direction == "cross_to_left":
                        cv2.arrowedLine(overlay, p2_s, p1_s, (0, 0, 255), line_thickness, tipLength=tip_length)
                    else:
                        cv2.line(overlay, p1_s, p2_s, (0, 0, 255), line_thickness)

            for track in frame_data.get('tracks', []):
                track_id = track['track_id']
                box = track['box_xyxy']
                x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])

                box_color = (128, 128, 128)
                alpha = 0.5
                if current_frame_index in event_frames_indices:
                    box_color = (128, 128, 128)
                    alpha = 0.8
                    if track.get('is_in_roi'):
                        box_color = (0, 255, 255)
                        alpha = 1.0
                    if track_id in active_alert_ids:
                        box_color = (0, 0, 255)
                        alpha = 1.0

                track_overlay = overlay.copy()
                cv2.rectangle(track_overlay, (x1, y1), (x2, y2), box_color, 2)
                anchor_coords = track.get('anchors', [])
                for anchor in anchor_coords:
                    center_point = (int(anchor[0] * scale_x), int(anchor[1] * scale_y))
                    cv2.circle(track_overlay, center_point, 5, box_color, -1)
                overlay = cv2.addWeighted(track_overlay, alpha, overlay, 1 - alpha, 0)
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

            if process.stdin: process.stdin.write(frame.tobytes())

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