# src/moshousapient/utils/video_utils.py

import subprocess
import json
import logging
import cv2
import os
from typing import List, Dict, Any


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
                logging.info(f"[系統] 已成功偵測到影片解析度: {width}x{height}")
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
    """
    從來源影片讀取指定幀，進行座標縮放後繪製追蹤框，並編碼成新的影片片段。
    """
    from ..settings import settings  # 需要 settings 來獲取分析解析度

    if not event_frames_data: return False

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        logging.error(f"無法開啟來源影片: {source_video_path}")
        return False

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps == 0:
        source_fps = 30.0
        logging.warning(f"無法讀取來源影片 FPS，將使用預設值: {source_fps}")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_x = source_width / settings.ANALYSIS_WIDTH
    scale_y = source_height / settings.ANALYSIS_HEIGHT

    frame_indices = sorted([f['frame_index'] for f in event_frames_data])
    start_frame = frame_indices[0]
    end_frame = frame_indices[-1]

    buffer_pre_frames = int(pre_event_sec * source_fps)
    buffer_post_frames = int(post_event_sec * source_fps)

    read_start_frame = max(1, start_frame - buffer_pre_frames)
    read_end_frame = end_frame + buffer_post_frames

    draw_data_map = {f['frame_index']: f['tracks'] for f in event_frames_data}

    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{source_width}x{source_height}', '-pix_fmt', 'bgr24', '-r', str(source_fps),
        '-i', '-', '-c:v', 'hevc_nvenc', '-preset', 'p6',
        '-r', str(output_fps),
        '-pix_fmt', 'yuv420p', output_path
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    logging.info(f"啟動 FFmpeg 為事件影片進行編碼: {os.path.basename(output_path)}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, read_start_frame - 1)
        for current_frame_index in range(read_start_frame, read_end_frame + 1):
            ret, frame = cap.read()
            if not ret: break

            if current_frame_index in draw_data_map:
                for track in draw_data_map[current_frame_index]:
                    box = track['box_xyxy']
                    x1 = int(box[0] * scale_x)
                    y1 = int(box[1] * scale_y)
                    x2 = int(box[2] * scale_x)
                    y2 = int(box[3] * scale_y)

                    track_id = track['track_id']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if process.stdin:
                process.stdin.write(frame.tobytes())
    except (BrokenPipeError, IOError):
        logging.warning("[FFmpeg] 管道已關閉。")
    finally:
        cap.release()
        _, stderr = process.communicate()
        if process.returncode != 0:
            logging.error(f"[FFmpeg] 編碼失敗:\n{stderr.decode('utf-8', errors='ignore')}")
            return False

    logging.info(f"事件影片已成功儲存: {os.path.basename(output_path)}")
    return True