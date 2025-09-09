# components/event_processor.py
import cv2
import time
import threading
import logging
import subprocess
import os
import pickle
from collections import deque
from queue import Empty, Queue
from datetime import datetime
import numpy as np
from ultralytics import YOLO
from config import Config
from database import SessionLocal
from models import Event
from utils.reid_utils import find_or_create_person


def inference_worker(frame_queue: Queue, shared_state: dict, stop_event: threading.Event,
                     lock: threading.Lock, model: YOLO, reid_model: YOLO, tracker):
    """
    AI 推論執行緒
    """
    logging.info("推論器: 執行緒已啟動, 使用 GPU。")

    frame_counter = 0
    reid_interval = 5

    latency_buffer, det_time_buffer, track_time_buffer, reid_time_buffer = [], [], [], []
    logging_interval_frames = 30

    while not stop_event.is_set():
        try:
            with lock:
                if shared_state.get('event_ended', False):
                    tracker.reset()
                    shared_state['event_ended'] = False
                    logging.info("[推論器] 偵測到事件結束，已重置追蹤器狀態。")

            item = frame_queue.get(timeout=1)
            frame_counter += 1

            t_capture = item['time']
            frame_low_res = cv2.resize(item['frame'], (Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT))

            t_det_start = time.time()
            dets_results = model(frame_low_res, device=0, verbose=False, classes=[0], conf=0.4)
            t_track_start = time.time()

            boxes_on_cpu = dets_results[0].boxes.cpu()
            tracks = tracker.update(boxes_on_cpu, frame_low_res)
            t_reid_start = time.time()

            reid_features_map = {}
            person_crops = []

            if len(tracks) > 0 and (frame_counter % reid_interval == 0):
                track_ids = tracks[:, 4].astype(int)
                xyxy_coords = tracks[:, :4]
                valid_track_ids = []
                for i, xyxy in enumerate(xyxy_coords):
                    x1, y1, x2, y2 = map(int, xyxy)
                    crop = frame_low_res[y1:y2, x1:x2]
                    if crop.size > 0:
                        person_crops.append(crop)
                        valid_track_ids.append(track_ids[i])

                if person_crops:
                    embeddings = reid_model.embed(person_crops, verbose=False)
                    for i, track_id in enumerate(valid_track_ids):
                        reid_features_map[track_id] = embeddings[i].cpu().numpy()

            t_reid_end = time.time()

            with lock:
                shared_state['person_detected'] = len(tracks) > 0
                shared_state['tracked_objects'] = tracks
                shared_state['reid_features_map'] = reid_features_map

            total_latency_ms = (t_reid_end - t_capture) * 1000
            det_time_ms = (t_track_start - t_det_start) * 1000
            track_time_ms = (t_reid_start - t_track_start) * 1000
            reid_time_ms = 0
            if person_crops:
                reid_time_ms = (t_reid_end - t_reid_start) * 1000

            latency_buffer.append(total_latency_ms)
            det_time_buffer.append(det_time_ms)
            track_time_buffer.append(track_time_ms)
            reid_time_buffer.append(reid_time_ms)

            if len(latency_buffer) >= logging_interval_frames:
                logging.info(
                    f"[推論器] 延遲統計 (avg over {len(latency_buffer)} frames): "
                    f"Total: {np.mean(latency_buffer):.1f} ms | "
                    f"Detect: {np.mean(det_time_buffer):.1f} ms, "
                    f"Track: {np.mean(track_time_buffer):.1f} ms, "
                    f"Re-ID: {np.mean(reid_time_buffer):.1f} ms"
                )
                latency_buffer.clear()
                det_time_buffer.clear()
                track_time_buffer.clear()
                reid_time_buffer.clear()

        except Empty:
            with lock:
                shared_state['person_detected'] = False
                shared_state['tracked_objects'] = np.empty((0, 5))
                shared_state['reid_features_map'] = {}
            continue
        except Exception as e:
            logging.error(f"推論器: 執行緒發生錯誤: {e}", exc_info=True)

    logging.info("推論器: 執行緒已停止。")

def frame_consumer(frame_queue: Queue, shared_state: dict, stop_event: threading.Event,
                   notifier, lock: threading.Lock, active_recorders: list):
    logging.info("消費者: 執行緒已啟動。")
    is_capturing_event, last_person_seen_time, last_event_ended_time = False, 0, 0
    buffer_size = int(Config.PRE_EVENT_SECONDS * Config.TARGET_FPS * 1.5)
    frame_buffer = deque(maxlen=buffer_size)
    event_recording, current_event_features = [], []

    debug_log_counter = 0
    DEBUG_LOG_INTERVAL = 60

    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=1)
            current_time = item['time']
            debug_log_counter += 1

            with lock:
                person_detected_now = shared_state.get('person_detected', False)
                num_tracks = len(shared_state.get('tracked_objects', []))
                reid_features_map_now = shared_state.get('reid_features_map', {})

            if debug_log_counter % DEBUG_LOG_INTERVAL == 0:
                logging.info(f"[消費者-偵錯] person_detected: {person_detected_now}, num_tracks: {num_tracks}, is_capturing: {is_capturing_event}")

            frame_data = {'frame': item['frame'], 'time': current_time, 'tracks': shared_state.get('tracked_objects', np.empty((0, 5)))}
            if is_capturing_event:
                event_recording.append(frame_data)
                if reid_features_map_now: current_event_features.extend(reid_features_map_now.values())
            else:
                frame_buffer.append(frame_data)
            if person_detected_now:
                last_person_seen_time = current_time
            if person_detected_now and not is_capturing_event:
                if current_time - last_event_ended_time > Config.COOLDOWN_PERIOD:
                    logging.info(">>> [消費者] 偵測到人物! 觸發事件錄製... <<<")
                    is_capturing_event = True
                    event_recording = list(frame_buffer)
                    current_event_features.clear()
                    if reid_features_map_now: current_event_features.extend(reid_features_map_now.values())
            if is_capturing_event and (current_time - last_person_seen_time > Config.POST_EVENT_SECONDS):
                logging.info("[消費者] 事件後續幀捕捉完成。")
                if len(event_recording) > 1:
                    duration = event_recording[-1]['time'] - event_recording[0]['time']
                    actual_fps = len(event_recording) / duration if duration > 0 else Config.TARGET_FPS
                    encoding_thread = threading.Thread(target=encode_and_send_video, name="EncodingThread", args=(list(event_recording), notifier, actual_fps, list(current_event_features)))
                    active_recorders.append(encoding_thread)
                    encoding_thread.start()
                is_capturing_event, event_recording = False, []
                current_event_features.clear()
                last_event_ended_time = current_time
                with lock: shared_state['event_ended'] = True
        except Empty:
            if is_capturing_event:
                logging.info("[消費者] 佇列為空，結束當前事件錄製。")
                if len(event_recording) > 1:
                    duration = event_recording[-1]['time'] - event_recording[0]['time']
                    actual_fps = len(event_recording) / duration if duration > 0 else Config.TARGET_FPS
                    encoding_thread = threading.Thread(target=encode_and_send_video, name="EncodingThread", args=(list(event_recording), notifier, actual_fps, list(current_event_features)))
                    active_recorders.append(encoding_thread)
                    encoding_thread.start()
                is_capturing_event, event_recording = False, []
                current_event_features.clear()
                last_event_ended_time = time.time()
                with lock: shared_state['event_ended'] = True
            continue
        except Exception as e:
            logging.error(f"消費者: 執行緒發生錯誤: {e}", exc_info=True)
    logging.info("消費者: 執行緒已停止。")



def encode_and_send_video(frame_data_list: list, notifier_instance, actual_fps: float, reid_features_list: list):
    """
    影片編碼執行緒: 使用 FFmpeg (NVENC) 硬體編碼, 處理 Re-ID 特徵, 儲存事件紀錄, 並透過 Notifier 發送。
    """
    if not frame_data_list or actual_fps <= 0:
        logging.warning("[編碼器] 沒有影像幀或無效的 FPS, 取消編碼。")
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

    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', frame_size_str,
        '-pix_fmt', 'bgr24', '-r', str(actual_fps), '-i', '-',
        '-c:v', 'hevc_nvenc', '-preset', 'p6', '-rc', 'vbr', '-cq', '30',
        '-b:v', '1M', '-maxrate', '2M', '-pix_fmt', 'yuv420p', save_path
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    scale_x = Config.ENCODE_WIDTH / Config.ANALYSIS_WIDTH
    scale_y = Config.ENCODE_HEIGHT / Config.ANALYSIS_HEIGHT

    try:
        for frame_data in frame_data_list:
            frame = frame_data['frame'].copy()
            tracked_objects = frame_data['tracks']

            for d in tracked_objects:
                x1, y1, x2, y2, track_id = d[:5]

                x1_s, y1_s = int(x1 * scale_x), int(y1 * scale_y)
                x2_s, y2_s = int(x2 * scale_x), int(y2 * scale_y)
                track_id = int(track_id)

                cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 2)
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            process.stdin.write(frame.tobytes())

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

    serialized_features = None
    mean_feature = None
    if reid_features_list:
        try:
            features_np = np.array(reid_features_list)
            mean_feature = np.mean(features_np, axis=0)
            serialized_features = pickle.dumps(mean_feature)
            logging.info(f"[特徵處理] 已成功將 {len(reid_features_list)} 個特徵向量聚合成一個平均特徵。")
        except Exception as e:
            logging.error(f"[特徵處理] 處理 Re-ID 特徵時發生錯誤: {e}", exc_info=True)

    if process.returncode != 0:
        logging.error(f"[GPU 編碼器] 錯誤: FFmpeg 返回非零退出碼: {process.returncode}")
        logging.error(f"[GPU 編碼器 stderr]: {stderr_output.decode('utf-8', errors='ignore').strip()}")
    else:
        logging.info(f"[資訊] 事件影片已儲存至: {save_path}")
        db = SessionLocal()
        try:
            person_id_for_event = None
            if mean_feature is not None:
                person_id, is_new = find_or_create_person(db, mean_feature)
                person_id_for_event = person_id
                if is_new:
                    logging.info(f"[Re-ID] 發現新的人物！已在全域畫廊中建立永久 ID: {person_id}")
                else:
                    logging.info(f"[Re-ID] 識別出已知人物。匹配到永久 ID: {person_id}")
            new_event = Event(
                video_path=save_path,
                event_type="person_detected",
                status="unreviewed",
                reid_features=serialized_features,
                person_id=person_id_for_event
            )
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
        notification_message = (
            f"**事件警報!**\n"
            f"於 `{timestamp_for_display}` 偵測到活動。\n"
            f"影片: {Config.ENCODE_HEIGHT}p @ {actual_fps:.1f} FPS, 編碼速率 {encoding_fps:.1f} FPS"
        )
        notifier_instance.schedule_notification(notification_message, file_path=save_path)