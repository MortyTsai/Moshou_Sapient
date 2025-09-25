# src/moshousapient/services/video_recorder.py

import logging
import os
import subprocess
from datetime import datetime
import numpy as np
import cv2

from ..config import Config
from .database_service import process_reid_and_identify_person, save_event
from ..settings import settings  # 新增


def encode_and_send_video(
        frame_data_list: list,
        notifier_instance,
        actual_fps: float,
        reid_features_list: list,
        event_type: str = "person_detected",
        video_fps_mode: str = "SOURCE",
        target_fps: float = 30.0
):
    if not frame_data_list or actual_fps <= 0:
        logging.warning("[編碼器] 沒有影像幀或無效的 FPS，取消編碼。")
        return

    if video_fps_mode == "TARGET" and target_fps > 0:
        output_fps = target_fps
        step = max(1, round(actual_fps / output_fps))
        sampled_frame_data_list = frame_data_list[::step]
    else:
        output_fps = actual_fps
        sampled_frame_data_list = frame_data_list

    num_frames = len(sampled_frame_data_list)
    logging.info(
        f">>> [GPU 編碼器] 收到 {num_frames} 幀影像 (事件類型: {event_type})，開始以 {output_fps:.2f} FPS 進行硬體編碼...")

    now = datetime.now()
    timestamp_for_filename = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{event_type}_{timestamp_for_filename}.mp4"
    save_path = os.path.join(Config.CAPTURES_DIR, filename)
    frame_size_str = f'{Config.ENCODE_WIDTH}x{Config.ENCODE_HEIGHT}'

    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', frame_size_str,
        '-pix_fmt', 'bgr24', '-r', str(output_fps), '-i', '-',
        '-c:v', 'hevc_nvenc', '-preset', 'p6'
    ]
    if Config.VIDEO_ENCODING_MODE == "BALANCED":
        bitrate_str = f"{Config.TARGET_BITRATE_MBPS}M"
        command.extend(['-rc', 'cbr', '-b:v', bitrate_str, '-maxrate', bitrate_str])
    else:
        quality_level = '30'
        command.extend(['-rc', 'vbr', '-cq', quality_level, '-b:v', '0', '-maxrate', '10M'])
    command.extend(['-pix_fmt', 'yuv420p', save_path])

    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    active_alert_ids = set()
    scale_x = Config.ENCODE_WIDTH / settings.ANALYSIS_WIDTH
    scale_y = Config.ENCODE_HEIGHT / settings.ANALYSIS_HEIGHT

    try:
        for frame_data in sampled_frame_data_list:
            frame = frame_data['frame'].copy()

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
                        cv2.arrowedLine(overlay, p1_s, p2_s, (0, 0, 255), line_thickness, tipLength=tip_length)
                        cv2.arrowedLine(overlay, p2_s, p1_s, (0, 0, 255), line_thickness, tipLength=tip_length)
            frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

            current_frame_track_ids = {int(t[4]) for t in frame_data.get('tracks', [])}
            for track_id in frame_data.get('tripwire_alert_ids', set()):
                active_alert_ids.add(track_id)
            active_alert_ids.intersection_update(current_frame_track_ids)

            for track in frame_data.get('tracks', []):
                box, track_id = track[:4], int(track[4])
                x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])
                track_roi_status = frame_data.get('track_roi_status', {})
                is_in_roi = track_roi_status.get(track_id, False)

                box_color = (0, 255, 0)
                if is_in_roi: box_color = (0, 255, 255)
                if track_id in active_alert_ids: box_color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            if process.stdin:
                process.stdin.write(frame.tobytes())

    except (BrokenPipeError, IOError):
        logging.warning("[GPU 編碼器] 警告: FFmpeg 程序在寫入完成前已關閉管道。")
    finally:
        if process.stdin:
            process.stdin.close()

    stderr_output_bytes, _ = process.communicate()
    if process.returncode != 0:
        stderr_output = stderr_output_bytes.decode('utf-8', errors='ignore')
        logging.error(f"[GPU 編碼器] 錯誤: FFmpeg 返回非零退出碼: {process.returncode}\n{stderr_output}")
        return

    logging.info(f"[資訊] 事件影片已儲存至: {save_path}")

    person_id = process_reid_and_identify_person(reid_features_list)
    save_event(save_path, event_type, person_id)

    if notifier_instance:
        timestamp_for_display = now.strftime("%Y-%m-%d %H:%M:%S")
        notification_message = f"**事件警報! ({event_type})**\n`{timestamp_for_display}` 偵測到活動。"
        notifier_instance.schedule_notification(notification_message, file_path=save_path)