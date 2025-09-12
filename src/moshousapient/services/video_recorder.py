# src/moshousapient/services/video_recorder.py

"""
事件影片處理器

此模組包含處理已觸發事件的後續動作，主要職責包括：
1. 使用 NVIDIA GPU 硬體加速，將事件期間的影像幀編碼為 MP4 影片檔案。
2. 在影片上繪製視覺化疊加層（如追蹤框、ROI 區域、警戒線）。
3. 執行 Re-ID 特徵聚類與比對，識別事件中的人物。
4. 將事件記錄寫入資料庫。
5. 透過通知器（如 Discord）發送警報。
"""

import logging
import os
import pickle
import subprocess
import time
from datetime import datetime

import cv2
from sqlalchemy.orm import selectinload

from moshousapient.config import Config
from moshousapient.database import SessionLocal
from moshousapient.models import Event, Person, PersonFeature
from moshousapient.utils.reid_utils import find_best_match_in_gallery, cosine_similarity


def encode_and_send_video(
        frame_data_list: list,
        notifier_instance,
        actual_fps: float,
        reid_features_list: list,
        event_type: str = "person_detected"
):
    """
    對指定的影像幀列表進行編碼、儲存，並處理 Re-ID 和通知。
    """
    if not frame_data_list or actual_fps <= 0:
        logging.warning("[編碼器] 沒有影像幀或無效的 FPS，取消編碼。")
        return

    num_frames = len(frame_data_list)
    logging.info(
        f">>> [GPU 編碼器] 收到 {num_frames} 幀影像 (事件類型: {event_type})，"
        f"開始以 {actual_fps:.2f} FPS 進行硬體編碼..."
    )

    start_time = time.time()
    now = datetime.now()
    timestamp_for_display = now.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_for_filename = now.strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"{event_type}_{timestamp_for_filename}.mp4"
    save_path = os.path.join(Config.CAPTURES_DIR, filename)
    frame_size_str = f'{Config.ENCODE_WIDTH}x{Config.ENCODE_HEIGHT}'

    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', frame_size_str,
        '-pix_fmt', 'bgr24', '-r', str(actual_fps), '-i', '-', '-c:v', 'hevc_nvenc',
        '-preset', 'p6', '-rc', 'vbr', '-cq', '30', '-b:v', '1M', '-maxrate', '2M',
        '-pix_fmt', 'yuv420p', save_path
    ]

    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    scale_x = Config.ENCODE_WIDTH / Config.ANALYSIS_WIDTH
    scale_y = Config.ENCODE_HEIGHT / Config.ANALYSIS_HEIGHT

    try:
        for frame_data in frame_data_list:
            frame = frame_data['frame'].copy()

            if Config.TRIPWIRE_LINE_OBJECTS:
                for tripwire_obj in Config.TRIPWIRE_LINE_OBJECTS:
                    line = tripwire_obj["line"]
                    p1, p2 = line.coords
                    p1_scaled = (int(p1[0] * scale_x), int(p1[1] * scale_y))
                    p2_scaled = (int(p2[0] * scale_x), int(p2[1] * scale_y))
                    cv2.arrowedLine(frame, p1_scaled, p2_scaled, (0, 0, 255), 2, tipLength=0.05)

            tracked_objects = frame_data['tracks']
            track_roi_status = frame_data.get('track_roi_status', {})
            alert_ids = frame_data.get('tripwire_alert_ids', set())

            for d in tracked_objects:
                x1, y1, x2, y2, track_id = d[:5]
                track_id = int(track_id)
                color = (255, 0, 255) if track_id in alert_ids else (
                    (0, 0, 255) if track_roi_status.get(track_id, False) else (0, 255, 0))
                x1_s, y1_s, x2_s, y2_s = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            process.stdin.write(frame.tobytes())

    except (BrokenPipeError, IOError):
        logging.warning("[GPU 編碼器] 警告: FFmpeg 程序在寫入完成前已關閉管道。")
    finally:
        if process.stdin:
            process.stdin.close()

    stdout_output, stderr_output = process.communicate()
    end_time = time.time()
    encoding_duration = end_time - start_time
    encoding_fps = num_frames / encoding_duration if encoding_duration > 0 else 0
    logging.info(
        f"[GPU 編碼器] FFmpeg 硬體編碼耗時: {encoding_duration:.2f} 秒。實際平均編碼幀率: {encoding_fps:.2f} FPS")

    event_person_id_val = _process_reid_and_db(reid_features_list)

    if process.returncode != 0:
        logging.error(
            f"[GPU 編碼器] 錯誤: FFmpeg 返回非零退出碼: {process.returncode}\n{stderr_output.decode('utf-8', errors='ignore').strip()}")
    else:
        logging.info(f"[資訊] 事件影片已儲存至: {save_path}")
        _save_event_to_db(save_path, event_type, event_person_id_val)
        if notifier_instance:
            notification_message = (
                f"**事件警報! ({event_type})**\n"
                f"於 `{timestamp_for_display}` 偵測到活動。\n"
                f"影片: {Config.ENCODE_HEIGHT}p @ {actual_fps:.1f} FPS, 編碼速率 {encoding_fps:.1f} FPS"
            )
            notifier_instance.schedule_notification(notification_message, file_path=save_path)


def _process_reid_and_db(reid_features_list: list) -> int | None:
    """處理 Re-ID 特徵聚類、資料庫比對，並返回主要人物的 ID。"""
    if not reid_features_list:
        return None

    unique_features = list({feat.tobytes(): feat for feat in reid_features_list}.values())
    logging.info(f"[特徵處理] 原始特徵數: {len(reid_features_list)}, 去重後: {len(unique_features)}")

    db = SessionLocal()
    try:
        event_clusters = []
        for feature in unique_features:
            best_match_cluster, highest_sim = None, -1.0
            for cluster in event_clusters:
                rep_feature = pickle.loads(cluster.features[0].feature)
                sim = cosine_similarity(feature, rep_feature)
                if sim > highest_sim:
                    highest_sim, best_match_cluster = sim, cluster

            if highest_sim >= Config.PERSON_MATCH_THRESHOLD and best_match_cluster:
                best_match_cluster.features.append(PersonFeature(feature=pickle.dumps(feature)))
            else:
                new_cluster = Person()
                new_cluster.features.append(PersonFeature(feature=pickle.dumps(feature)))
                event_clusters.append(new_cluster)

        logging.info(f"[特徵處理] 事件內聚類完成，發現 {len(event_clusters)} 個潛在獨立人物。")

        static_persons_gallery = db.query(Person).options(selectinload(Person.features)).all()
        initial_db_persons = set(static_persons_gallery)

        final_person_map = {}
        for cluster in event_clusters:
            rep_feature = pickle.loads(cluster.features[0].feature)
            db_match = find_best_match_in_gallery(rep_feature, static_persons_gallery)
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

        if not event_clusters:
            return None

        first_person_obj = final_person_map[event_clusters[0]]
        db.flush()
        event_person_id_val = first_person_obj.id
        db.commit()

        num_new = len(unique_persons_in_event - initial_db_persons)
        num_reid = len(unique_persons_in_event.intersection(initial_db_persons))
        logging.info(
            f"[資料庫] Re-ID 處理完成。本次事件涉及 {len(unique_persons_in_event)} 人 (新增 {num_new}, 識別 {num_reid})。")

        return event_person_id_val

    except Exception as e:
        logging.error(f"[特徵處理] 處理 Re-ID 時發生錯誤，交易已回滾: {e}", exc_info=True)
        db.rollback()
        return None
    finally:
        db.close()


def _save_event_to_db(save_path: str, event_type: str, person_id: int | None):
    """將事件本身儲存到資料庫。"""
    db = SessionLocal()
    try:
        new_event = Event(video_path=save_path, event_type=event_type, status="unreviewed", person_id=person_id)
        db.add(new_event)
        db.commit()
        logging.info(f"[資料庫] 已成功將事件紀錄 (ID: {new_event.id}) 寫入資料庫。")
    except Exception as e:
        logging.error(f"[資料庫] 寫入事件紀錄時發生錯誤: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()