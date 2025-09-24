# src/moshousapient/services/video_recorder.py

import logging
import os
import subprocess
from datetime import datetime

from ..config import Config
from .database_service import process_reid_and_identify_person, save_event


def encode_and_send_video(
        frame_data_list: list,
        notifier_instance,
        actual_fps: float,
        reid_features_list: list,
        event_type: str = "person_detected",
        video_fps_mode: str = "SOURCE",
        target_fps: float = 30.0
):
    """對指定的影像幀列表進行編碼、儲存，並處理 Re-ID 和通知。"""
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
    timestamp_for_filename = now.strftime("%Y-%m-%d_%H-%M-%S")
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
    else:  # QUALITY
        quality_level = '30'
        command.extend(['-rc', 'vbr', '-cq', quality_level, '-b:v', '0', '-maxrate', '10M'])
    command.extend(['-pix_fmt', 'yuv4p', save_path])

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    try:
        for frame_data in sampled_frame_data_list:
            frame = frame_data['frame'].copy()
            # ... (此處可以添加繪圖邏輯，如果需要的話) ...
            process.stdin.write(frame.tobytes())
    except (BrokenPipeError, IOError):
        logging.warning("[GPU 編碼器] 警告: FFmpeg 程序在寫入完成前已關閉管道。")
    finally:
        if process.stdin:
            process.stdin.close()

    _, stderr_output = process.communicate()
    return_code = process.wait()

    if return_code != 0:
        logging.error(
            f"[GPU 編碼器] 錯誤: FFmpeg 返回非零退出碼: {return_code}\n{stderr_output.decode('utf-8', errors='ignore').strip()}")
        return

    logging.info(f"[資訊] 事件影片已儲存至: {save_path}")

    person_id = process_reid_and_identify_person(reid_features_list)
    save_event(save_path, event_type, person_id)

    if notifier_instance:
        timestamp_for_display = now.strftime("%Y-%m-%d %H:%M:%S")
        notification_message = (
            f"**事件警報! ({event_type})**\n"
            f"`{timestamp_for_display}` 偵測到活動。"
        )
        notifier_instance.schedule_notification(notification_message, file_path=save_path)