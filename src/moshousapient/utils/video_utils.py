# src/moshousapient/utils/video_utils.py

import subprocess
import json
import logging
import cv2
import os
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
        output_fps: int,
        pre_event_sec: float,
        post_event_sec: float
) -> bool:
    if not event_frames_data:
        return False

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        logging.error(f"無法開啟來源影片: {source_video_path}")
        return False

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_x = source_width / settings.ANALYSIS_WIDTH
    scale_y = source_height / settings.ANALYSIS_HEIGHT

    frame_indices = sorted([f['frame_index'] for f in event_frames_data])
    start_frame, end_frame = frame_indices[0], frame_indices[-1]

    buffer_pre_frames = int(pre_event_sec * source_fps)
    buffer_post_frames = int(post_event_sec * source_fps)

    read_start_frame = max(1, start_frame - buffer_pre_frames)
    read_end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), end_frame + buffer_post_frames)

    draw_data_map = {f['frame_index']: f['tracks'] for f in event_frames_data}

    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{source_width}x{source_height}',
        '-pix_fmt', 'bgr24', '-r', str(source_fps),
        '-i', '-',
        '-c:v', 'hevc_nvenc', '-preset', 'p6',
        '-r', str(output_fps),
        '-pix_fmt', 'yuv420p', output_path
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    logging.info(f"啟動 FFmpeg 為事件影片進行編碼: {os.path.basename(output_path)}")

    active_alert_ids = set()

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, read_start_frame - 1)
        for current_frame_index in range(read_start_frame, read_end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            overlay = frame.copy()

            if Config.ROI_ENABLED and Config.ROI_POLYGON_OBJECT:
                roi_points = np.array(Config.ROI_POLYGON_OBJECT.exterior.coords, dtype=np.int32)
                roi_points_scaled = (roi_points * np.array([scale_x, scale_y])).astype(np.int32)
                cv2.fillPoly(overlay, [roi_points_scaled], color=(255, 255, 0))
                cv2.polylines(overlay, [roi_points_scaled], isClosed=True, color=(255, 255, 0), thickness=4)

            if Config.TRIPWIRES_ENABLED and Config.TRIPWIRE_LINE_OBJECTS:
                line_thickness = 8  # 增加線條粗細
                tip_length = 0.02  # 箭頭尖端相對於線條長度的比例

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
                        cv2.arrowedLine(overlay, p1_s, p2_s, (0, 0, 255), line_thickness, tipLength=tip_length)
                        cv2.arrowedLine(overlay, p2_s, p1_s, (0, 0, 255), line_thickness, tipLength=tip_length)

            frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

            current_frame_track_ids = set()
            if current_frame_index in draw_data_map:
                for track in draw_data_map[current_frame_index]:
                    track_id = track['track_id']
                    current_frame_track_ids.add(track_id)
                    if track.get('has_crossed_tripwire'):
                        active_alert_ids.add(track_id)

            active_alert_ids.intersection_update(current_frame_track_ids)

            if current_frame_index in draw_data_map:
                for track in draw_data_map[current_frame_index]:
                    box = track['box_xyxy']
                    x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])
                    track_id = track['track_id']

                    box_color = (0, 255, 0)
                    if track.get('is_in_roi', False): box_color = (0, 255, 255)
                    if track_id in active_alert_ids: box_color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            text_position = (20, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_color = (255, 255, 255)
            thickness = 2

            if current_frame_index < start_frame:
                time_left = (start_frame - current_frame_index) / source_fps
                if time_left > 0:
                    text = f"Pre-Event Buffer: {time_left:.1f}s"
                    cv2.putText(frame, text, text_position, font, font_scale, font_color, thickness, cv2.LINE_AA)
            elif current_frame_index > end_frame:
                time_left = post_event_sec - (current_frame_index - end_frame) / source_fps
                if time_left > 0:
                    text = f"Post-Event Buffer: {time_left:.1f}s"
                    cv2.putText(frame, text, text_position, font, font_scale, font_color, thickness, cv2.LINE_AA)

            if process.stdin:
                process.stdin.write(frame.tobytes())

    except (BrokenPipeError, IOError):
        logging.warning("[FFmpeg] 管道提前關閉。")
    finally:
        cap.release()
        if process.stdin:
            process.stdin.close()

    stderr_output_bytes, _ = process.communicate()

    if process.returncode != 0:
        stderr_output = stderr_output_bytes.decode('utf-8', errors='ignore')
        logging.error(f"[FFmpeg] 編碼失敗 (返回碼: {process.returncode}):\n{stderr_output}")
        return False

    logging.info(f"事件影片已成功儲存: {os.path.basename(output_path)}")
    return True