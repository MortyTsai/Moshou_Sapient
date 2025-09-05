# components/event_processor.py
import cv2
import time
import threading
import logging
import subprocess
import os
import numpy as np
from collections import deque
from queue import Empty, Queue
from datetime import datetime
from typing import Dict

from ultralytics import YOLO
from config import Config
from components.trackers.sort_tracker import Sort
from database import SessionLocal
from models import Event


def inference_worker(frame_queue: Queue, shared_state: dict, stop_event: threading.Event,
                     lock: threading.Lock, model: YOLO, class_names: Dict[int, str], tracker: Sort):
    """
    AI推論執行緒: 從佇列中獲取影像, 進行物件偵測與追蹤, 並更新共享狀態。
    """
    logging.info("推論器: 執行緒已啟動, 使用 GPU。")
    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=1)
            frame_low_res = cv2.resize(item['frame'], (Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT))

            results = model.predict(frame_low_res, device=0, verbose=False, max_det=10)

            detections_list = []
            for r in results:
                for box in r.boxes:
                    if class_names[int(box.cls[0])] == 'person' and box.conf[0] > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections_list.append([x1, y1, x2, y2, conf])

            detections_np = np.array(detections_list) if detections_list else np.empty((0, 5))
            tracked_objects = tracker.update(detections_np)

            with lock:
                shared_state['person_detected'] = len(tracked_objects) > 0
                shared_state['tracked_objects'] = tracked_objects

        except Empty:
            with lock:
                tracked_objects = tracker.update(np.empty((0, 5)))
                shared_state['person_detected'] = len(tracked_objects) > 0
                shared_state['tracked_objects'] = tracked_objects
            continue
        except Exception as e:
            logging.error(f"推論器: 執行緒發生錯誤: {e}", exc_info=True)
    logging.info("推論器: 執行緒已停止。")


def frame_consumer(frame_queue: Queue, shared_state: dict, stop_event: threading.Event,
                   notifier, lock: threading.Lock):
    """
    消費者執行緒: 處理事件邏輯、觸發錄影, 並啟動編碼執行緒。
    """
    logging.info("消費者: 執行緒已啟動。")
    is_capturing_event = False
    last_person_seen_time = 0
    last_event_ended_time = 0

    buffer_size = int(Config.PRE_EVENT_SECONDS * Config.TARGET_FPS * 1.5)
    frame_buffer = deque(maxlen=buffer_size)
    event_recording = []

    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=1)
            current_time = item['time']

            with lock:
                person_detected_now = shared_state.get('person_detected', False)
                tracked_objects_now = shared_state.get('tracked_objects', np.empty((0, 5)))

            frame_data = {'frame': item['frame'], 'time': current_time, 'tracks': tracked_objects_now}

            if is_capturing_event:
                event_recording.append(frame_data)
            else:
                frame_buffer.append(frame_data)

            if person_detected_now:
                last_person_seen_time = current_time

            if person_detected_now and not is_capturing_event:
                if current_time - last_event_ended_time > Config.COOLDOWN_PERIOD:
                    logging.info(">>> [消費者] 偵測到人物! 觸發事件錄製... <<<")
                    is_capturing_event = True
                    event_recording = list(frame_buffer)

            if is_capturing_event and (current_time - last_person_seen_time > Config.POST_EVENT_SECONDS):
                logging.info("[消費者] 事件後續幀捕捉完成。")
                if len(event_recording) > 1:
                    duration = event_recording[-1]['time'] - event_recording[0]['time']
                    actual_fps = len(event_recording) / duration if duration > 0 else Config.TARGET_FPS

                    encoding_thread = threading.Thread(target=encode_and_send_video,
                                                       name="EncodingThread",
                                                       args=(list(event_recording), notifier, actual_fps))
                    encoding_thread.start()

                is_capturing_event, event_recording = False, []
                last_event_ended_time = current_time

        except Empty:
            continue
        except Exception as e:
            logging.error(f"消費者: 執行緒發生錯誤: {e}", exc_info=True)
    logging.info("消費者: 執行緒已停止。")


def encode_and_send_video(frame_data_list: list, notifier_instance, actual_fps: float):
    """
    影片編碼執行緒: 使用 FFmpeg (NVENC) 硬體編碼, 並透過 Notifier 發送。
    """
    if not frame_data_list or actual_fps <= 0:
        logging.warning("[編碼器] 沒有影像幀或無效的FPS, 取消編碼。")
        return

    num_frames = len(frame_data_list)
    logging.info(f">>> [GPU 編碼器] 收到 {num_frames} 幀影像, 開始以 {actual_fps:.2f} FPS 進行硬體編碼...")
    start_time = time.time()
    now = datetime.now()
    timestamp_for_display = now.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_for_filename = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"event_{timestamp_for_filename}.mp4"
    save_path = os.path.join(Config.CAPTURES_DIR, filename)

    frame_size_str = f'{Config.ENCODE_WIDTH}x{Config.ENCODE_HEIGHT}'
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', frame_size_str, '-pix_fmt', 'bgr24',
               '-r', str(actual_fps), '-i', '-', '-c:v', 'hevc_nvenc', '-preset', 'p6', '-rc', 'vbr', '-cq', '30',
               '-b:v', '1M', '-maxrate', '2M', '-pix_fmt', 'yuv420p', save_path]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    scale_x = Config.ENCODE_WIDTH / Config.ANALYSIS_WIDTH
    scale_y = Config.ENCODE_HEIGHT / Config.ANALYSIS_HEIGHT

    try:
        for frame_data in frame_data_list:
            frame = frame_data['frame'].copy()
            tracked_objects = frame_data['tracks']

            for d in tracked_objects:
                x1, y1, x2, y2, track_id = d
                x1_s, y1_s = int(x1 * scale_x), int(y1 * scale_y)
                x2_s, y2_s = int(x2 * scale_x), int(y2 * scale_y)
                track_id = int(track_id)

                cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 2)
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            resized_frame = cv2.resize(frame, (Config.ENCODE_WIDTH, Config.ENCODE_HEIGHT))
            process.stdin.write(resized_frame.tobytes())
    except (BrokenPipeError, IOError):
        logging.warning("[GPU 編碼器] 警告: FFmpeg 程序在寫入完成前已關閉管道。")
    except Exception as e:
        logging.error(f"[GPU 編碼器] 錯誤: 寫入幀時發生例外: {e}", exc_info=True)
    finally:
        if process.stdin:
            process.stdin.close()
        stdout_output, stderr_output = process.communicate()
        end_time = time.time()

    encoding_duration = end_time - start_time
    encoding_fps = num_frames / encoding_duration if encoding_duration > 0 else 0
    logging.info(f"[GPU 編碼器] FFmpeg 硬體編碼耗時: {encoding_duration:.2f} 秒。")
    logging.info(f"[GPU 編碼器] ===> 實際平均編碼幀率: {encoding_fps:.2f} FPS <===")

    if process.returncode != 0:
        logging.error(f"[GPU 編碼器] 錯誤: FFmpeg 返回非零退出碼: {process.returncode}")
        logging.error(f"[GPU 編碼器 stderr]: {stderr_output.decode('utf-8', errors='ignore').strip()}")
    else:
        logging.info(f"[資訊] 事件影片已儲存至: {save_path}")

        db = SessionLocal()
        try:
            new_event = Event(video_path=save_path, event_type="person_detected", status="unreviewed")
            db.add(new_event)
            db.commit()
            db.refresh(new_event)
            logging.info(f"[資料庫] 已成功將事件紀錄 (ID: {new_event.id}) 寫入資料庫。")
        except Exception as e:
            logging.error(f"[資料庫] 寫入事件紀錄時發生錯誤: {e}", exc_info=True)
            db.rollback()
        finally:
            db.close()

        if notifier_instance:
            notification_message = (f"**事件警報!**\n"
                                    f"於 `{timestamp_for_display}` 偵測到活動。\n"
                                    f"影片: {Config.ENCODE_HEIGHT}p @ {actual_fps:.1f} FPS, 編碼速率 {encoding_fps:.1f} FPS")
            notifier_instance.schedule_notification(notification_message, file_path=save_path)