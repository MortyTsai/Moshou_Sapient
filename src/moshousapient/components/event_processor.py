# src/moshousapient/components/event_processor.py

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
from shapely.geometry import Point, LineString
from ..utils.geometry_utils import get_point_side_of_line
from ..config import Config
from ..database import SessionLocal
from ..models import Event, Person, PersonFeature
from ..utils.reid_utils import cosine_similarity, find_best_match_in_gallery
from sqlalchemy.orm import selectinload


def inference_worker(frame_queue: Queue, shared_state: dict, stop_event: threading.Event,
                     lock: threading.Lock, model: YOLO, reid_model: YOLO, tracker):
    """AI 推論執行緒"""
    logging.info("[推論器] 執行緒已啟動, 使用 GPU。")
    frame_counter = 0
    reid_interval = 5
    latency_buffer, det_time_buffer, track_time_buffer, reid_time_buffer = [], [], [], []
    logging_interval_frames = 60
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
            track_roi_status = {}
            if Config.ROI_POLYGON_OBJECT and len(tracks) > 0:
                for track in tracks:
                    x1, y1, x2, y2, track_id = track[:5]
                    bottom_center_point = Point((x1 + x2) / 2, y2)
                    track_roi_status[int(track_id)] = Config.ROI_POLYGON_OBJECT.contains(bottom_center_point)
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
                shared_state['track_roi_status'] = track_roi_status
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
                    f"[推論器] 延遲統計 (avg over {len(latency_buffer)} frames): Total: {np.mean(latency_buffer):.1f} ms | Detect: {np.mean(det_time_buffer):.1f} ms, Track: {np.mean(track_time_buffer):.1f} ms, Re-ID: {np.mean(reid_time_buffer):.1f} ms")
                latency_buffer.clear()
                det_time_buffer.clear()
                track_time_buffer.clear()
                reid_time_buffer.clear()
        except Empty:
            with lock:
                shared_state['person_detected'] = False
                shared_state['tracked_objects'] = np.empty((0, 5))
                shared_state['reid_features_map'] = {}
                shared_state['track_roi_status'] = {}
            continue
        except Exception as e:
            logging.error(f"[推論器] 執行緒發生錯誤: {e}", exc_info=True)
    logging.info("[推論器] 執行緒已停止。")

def frame_consumer(frame_queue: Queue, shared_state: dict, stop_event: threading.Event,
                   notifier, lock: threading.Lock, active_recorders: list):
    logging.info("[消費者] 執行緒已啟動。")
    is_capturing_event, last_person_seen_time, last_event_ended_time = False, 0, 0
    buffer_size = int(Config.PRE_EVENT_SECONDS * Config.TARGET_FPS * 1.5)
    frame_buffer = deque(maxlen=buffer_size)
    event_recording, current_event_features = [], []
    current_event_type = None
    dwell_time_trackers = {}
    track_last_positions = {}
    tripwire_alert_ids = set()

    debug_log_counter = 0
    DEBUG_LOG_INTERVAL = 60
    event_start_time = 0

    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=1)
            current_time = item['time']
            debug_log_counter += 1
            with lock:
                person_detected_now = shared_state.get('person_detected', False)
                current_tracks = shared_state.get('tracked_objects', [])
                reid_features_map_now = shared_state.get('reid_features_map', {})
                track_roi_status_now = shared_state.get('track_roi_status', {})

            if debug_log_counter % DEBUG_LOG_INTERVAL == 0:
                logging.info(
                    f"[消費者-偵錯] person_detected: {person_detected_now}, num_tracks: {len(current_tracks)}, is_capturing: {is_capturing_event}")

            current_tracked_ids = set()
            if len(current_tracks) > 0:
                for track in current_tracks:
                    x1, y1, x2, y2, track_id = track[:5]
                    track_id = int(track_id)
                    current_tracked_ids.add(track_id)

                    current_position = Point((x1 + x2) / 2, y2)
                    last_position = track_last_positions.get(track_id)

                    if last_position and last_position != current_position and Config.TRIPWIRE_LINE_OBJECTS:
                        movement_line = LineString([last_position, current_position])
                        for tripwire_obj in Config.TRIPWIRE_LINE_OBJECTS:
                            tripwire_line = tripwire_obj["line"]
                            alert_direction = tripwire_obj["direction"]

                            if movement_line.intersects(tripwire_line):
                                p1, p2 = tripwire_line.coords
                                side_before = get_point_side_of_line(last_position, Point(p1), Point(p2))
                                side_after = get_point_side_of_line(current_position, Point(p1), Point(p2))

                                if side_before != 0 and side_after != 0 and side_before != side_after:
                                    crossed_to_right = side_before == 1 and side_after == -1
                                    crossed_to_left = side_before == -1 and side_after == 1

                                    should_alert = (alert_direction == "both" or
                                                    (alert_direction == "cross_to_right" and crossed_to_right) or
                                                    (alert_direction == "cross_to_left" and crossed_to_left))

                                    if should_alert:
                                        logging.warning(
                                            f"--- [方向性警報] --- 目標 ID: {track_id} 觸發了警戒線！")
                                        tripwire_alert_ids.add(track_id)
                                        if not is_capturing_event:
                                            current_event_type = "tripwire_alert"
                                        elif current_event_type in ["person_detected", "dwell_alert"]:
                                            logging.info(
                                                f">>> [事件升級] '{current_event_type}' 事件已升級為 'tripwire_alert'。")
                                            current_event_type = "tripwire_alert"
                                        break

                    track_last_positions[track_id] = current_position

            # --- 核心修正：當一個 ID 消失時，同時清除其高亮狀態 ---
            disappeared_ids = set(track_last_positions.keys()) - current_tracked_ids
            for track_id in disappeared_ids:
                del track_last_positions[track_id]
                # 使用 .discard() 而不是 .remove() 以避免 track_id 不在集合中時引發錯誤
                tripwire_alert_ids.discard(track_id)

            if Config.ROI_POLYGON_OBJECT:
                for track_id, is_in_roi in track_roi_status_now.items():
                    if is_in_roi:
                        if track_id not in dwell_time_trackers:
                            dwell_time_trackers[track_id] = {'start_time': current_time, 'alerted': False}
                        else:
                            tracker_info = dwell_time_trackers[track_id]
                            if not tracker_info['alerted']:
                                dwell_duration = current_time - tracker_info['start_time']
                                if dwell_duration > Config.ROI_DWELL_TIME_THRESHOLD:
                                    logging.warning(
                                        f"--- [行為警報] --- 目標 ID: {track_id} 在 ROI 區域停留已超過 {Config.ROI_DWELL_TIME_THRESHOLD} 秒！")
                                    tracker_info['alerted'] = True
                                    if not is_capturing_event:
                                        current_event_type = "dwell_alert"
                                    elif current_event_type == "person_detected":
                                        logging.info(f">>> [事件升級] 'person_detected' 事件已升級為 'dwell_alert'。")
                                        current_event_type = "dwell_alert"
                    else:
                        if track_id in dwell_time_trackers:
                            del dwell_time_trackers[track_id]

                disappeared_ids_dwell = set(dwell_time_trackers.keys()) - current_tracked_ids
                for track_id in disappeared_ids_dwell:
                    del dwell_time_trackers[track_id]

            frame_data = {'frame': item['frame'], 'time': current_time,
                          'tracks': shared_state.get('tracked_objects', np.empty((0, 5))),
                          'track_roi_status': track_roi_status_now,
                          'tripwire_alert_ids': tripwire_alert_ids.copy()}

            if is_capturing_event:
                event_recording.append(frame_data)
                if reid_features_map_now:
                    current_event_features.extend(reid_features_map_now.values())
            else:
                frame_buffer.append(frame_data)

            if person_detected_now:
                last_person_seen_time = current_time

            if not is_capturing_event:
                if current_event_type in ["dwell_alert", "tripwire_alert"]:
                    logging.info(f">>> [消費者] 偵測到 '{current_event_type}' 事件! 觸發事件錄製... <<<")
                    is_capturing_event = True
                elif person_detected_now and current_time - last_event_ended_time > Config.COOLDOWN_PERIOD:
                    current_event_type = "person_detected"
                    logging.info(f">>> [消費者] 偵測到 '{current_event_type}' 事件! 觸發事件錄製... <<<")
                    is_capturing_event = True

                if is_capturing_event:
                    event_recording = list(frame_buffer)
                    event_start_time = current_time
                    current_event_features.clear()
                    tripwire_alert_ids.clear()

            should_end_event = False
            end_reason = ""
            if is_capturing_event:
                if not person_detected_now and current_time - last_person_seen_time > Config.POST_EVENT_SECONDS:
                    should_end_event = True
                    end_reason = "人物消失"
                elif current_time - event_start_time > Config.MAX_EVENT_DURATION:
                    should_end_event = True
                    end_reason = "超過最大錄影時長"

            if should_end_event:
                logging.info(f"[消費者] 事件結束 ({end_reason})。")
                if len(event_recording) > 1:
                    duration = event_recording[-1]['time'] - event_recording[0]['time']
                    actual_fps = len(event_recording) / duration if duration > 0 else Config.TARGET_FPS
                    encoding_thread = threading.Thread(target=encode_and_send_video, name="EncodingThread",
                                                       args=(list(event_recording), notifier, actual_fps,
                                                             list(current_event_features), current_event_type))
                    active_recorders.append(encoding_thread)
                    encoding_thread.start()
                is_capturing_event, event_recording = False, []
                current_event_features.clear()
                current_event_type = None
                last_event_ended_time = current_time
                with lock:
                    shared_state['event_ended'] = True
        except Empty:
            if is_capturing_event:
                logging.info("[消費者] 佇列為空，結束當前事件錄製。")
                if len(event_recording) > 1:
                    duration = event_recording[-1]['time'] - event_recording[0]['time']
                    actual_fps = len(event_recording) / duration if duration > 0 else Config.TARGET_FPS
                    encoding_thread = threading.Thread(target=encode_and_send_video, name="EncodingThread",
                                                       args=(list(event_recording), notifier, actual_fps,
                                                             list(current_event_features), current_event_type))
                    active_recorders.append(encoding_thread)
                    encoding_thread.start()
                is_capturing_event, event_recording = False, []
                current_event_features.clear()
                current_event_type = None
                last_event_ended_time = time.time()
                with lock:
                    shared_state['event_ended'] = True
            continue
        except Exception as e:
            logging.error(f"[消費者] 執行緒發生錯誤: {e}", exc_info=True)
    logging.info("[消費者] 執行緒已停止。")

def encode_and_send_video(frame_data_list: list, notifier_instance, actual_fps: float,
                          reid_features_list: list, event_type: str = "person_detected"):
    if not frame_data_list or actual_fps <= 0:
        logging.warning("[編碼器] 沒有影像幀或無效的 FPS, 取消編碼。")
        return

    num_frames = len(frame_data_list)
    logging.info(
        f">>> [GPU 編碼器] 收到 {num_frames} 幀影像 (事件類型: {event_type}), 開始以 "
        f"{actual_fps:.2f} FPS 進行硬體編碼...")

    start_time = time.time()
    now = datetime.now()
    timestamp_for_display = now.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_for_filename = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{event_type}_{timestamp_for_filename}.mp4"
    save_path = os.path.join(Config.CAPTURES_DIR, filename)
    frame_size_str = f'{Config.ENCODE_WIDTH}x{Config.ENCODE_HEIGHT}'

    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', frame_size_str,
               '-pix_fmt', 'bgr24', '-r', str(actual_fps), '-i', '-', '-c:v', 'hevc_nvenc',
               '-preset', 'p6', '-rc', 'vbr', '-cq', '30', '-b:v', '1M', '-maxrate', '2M',
               '-pix_fmt', 'yuv420p', save_path]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    scale_x = Config.ENCODE_WIDTH / Config.ANALYSIS_WIDTH
    scale_y = Config.ENCODE_HEIGHT / Config.ANALYSIS_HEIGHT

    try:
        for frame_data in frame_data_list:
            frame = frame_data['frame'].copy()

            if Config.TRIPWIRE_LINE_OBJECTS:
                for tripwire_obj in Config.TRIPWIRE_LINE_OBJECTS:
                    # --- 核心修正：從字典中正確獲取 line 物件 ---
                    line = tripwire_obj["line"]
                    p1, p2 = line.coords
                    p1_scaled = (int(p1[0] * scale_x), int(p1[1] * scale_y))
                    p2_scaled = (int(p2[0] * scale_x), int(p2[1] * scale_y))
                    # 繪製箭頭以表示方向
                    cv2.arrowedLine(frame, p1_scaled, p2_scaled, (0, 0, 255), 2, tipLength=0.05)

            tracked_objects = frame_data['tracks']
            track_roi_status = frame_data.get('track_roi_status', {})
            alert_ids = frame_data.get('tripwire_alert_ids', set())

            for d in tracked_objects:
                x1, y1, x2, y2, track_id = d[:5]
                track_id = int(track_id)

                if track_id in alert_ids:
                    color = (255, 0, 255)
                else:
                    is_in_roi = track_roi_status.get(track_id, False)
                    color = (0, 0, 255) if is_in_roi else (0, 255, 0)

                x1_s, y1_s = int(x1 * scale_x), int(y1 * scale_y)
                x2_s, y2_s = int(x2 * scale_x), int(y2 * scale_y)

                cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
    event_person_id_val = None
    if reid_features_list:
        unique_features = []
        seen_features_hashes = set()
        for feature in reid_features_list:
            feature_hash = feature.tobytes()
            if feature_hash not in seen_features_hashes:
                unique_features.append(feature)
                seen_features_hashes.add(feature_hash)
        logging.info(
            f"[特徵處理] 開始處理事件。原始特徵數: {len(reid_features_list)}, 去重後獨立特徵數: {len(unique_features)}")
        db = SessionLocal()
        try:
            event_clusters = []
            for feature in unique_features:
                best_match_cluster = None
                highest_sim = -1.0
                for cluster in event_clusters:
                    representative_feature = pickle.loads(cluster.features[0].feature)
                    sim = cosine_similarity(feature, representative_feature)
                    if sim > highest_sim:
                        highest_sim = sim
                        best_match_cluster = cluster
                if highest_sim >= Config.PERSON_MATCH_THRESHOLD and best_match_cluster:
                    new_feature_obj = PersonFeature(feature=pickle.dumps(feature))
                    best_match_cluster.features.append(new_feature_obj)
                else:
                    new_cluster = Person()
                    new_feature_obj = PersonFeature(feature=pickle.dumps(feature))
                    new_cluster.features.append(new_feature_obj)
                    event_clusters.append(new_cluster)
            logging.info(f"[特徵處理] 事件內聚類完成，發現 {len(event_clusters)} 個潛在獨立人物。")
            static_persons_gallery = db.query(Person).options(selectinload(Person.features)).all()
            initial_db_persons = set(static_persons_gallery)
            final_person_map = {}
            for cluster in event_clusters:
                representative_feature = pickle.loads(cluster.features[0].feature)
                db_match = find_best_match_in_gallery(representative_feature, static_persons_gallery)
                if db_match:
                    final_person_map[cluster] = db_match
                else:
                    db.add(cluster)
                    final_person_map[cluster] = cluster
                    static_persons_gallery.append(cluster)
            unique_persons_in_event = set(final_person_map.values())
            for cluster, final_person in final_person_map.items():
                if final_person != cluster:
                    for feature_obj in cluster.features:
                        final_person.features.append(PersonFeature(feature=feature_obj.feature))
                if final_person in initial_db_persons:
                    final_person.sighting_count += 1
            if event_clusters:
                first_cluster = event_clusters[0]
                first_person_obj = final_person_map[first_cluster]
                db.flush()
                event_person_id_val = first_person_obj.id
            db.commit()
            reidentified_persons = unique_persons_in_event.intersection(initial_db_persons)
            new_persons = unique_persons_in_event.difference(initial_db_persons)
            num_reidentified = len(reidentified_persons)
            num_new = len(new_persons)
            log_msg = (
                f"[資料庫] 成功提交 Re-ID 處理結果。本次事件涉及 {len(unique_persons_in_event)} 個獨立人物 (新增 {num_new} 人, 識別 {num_reidentified} 人)。")
            logging.info(log_msg)
        except Exception as e:
            logging.error(f"[特徵處理] 處理 Re-ID 特徵時發生嚴重錯誤，交易已回滾: {e}", exc_info=True)
            db.rollback()
            event_person_id_val = None
        finally:
            db.close()
    if process.returncode != 0:
        logging.error(f"[GPU 編碼器] 錯誤: FFmpeg 返回非零退出碼: {process.returncode}")
        logging.error(f"[GPU 編碼器 stderr]: {stderr_output.decode('utf-8', errors='ignore').strip()}")
    else:
        logging.info(f"[資訊] 事件影片已儲存至: {save_path}")
        db = SessionLocal()
        try:
            new_event = Event(video_path=save_path, event_type=event_type, status="unreviewed",
                              person_id=event_person_id_val)
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
            f"**事件警報! ({event_type})**\n" f"於 `{timestamp_for_display}` 偵測到活動。\n" f"影片: {Config.ENCODE_HEIGHT}p @ {actual_fps:.1f} FPS, 編碼速率 {encoding_fps:.1f} FPS")
        notifier_instance.schedule_notification(notification_message, file_path=save_path)

