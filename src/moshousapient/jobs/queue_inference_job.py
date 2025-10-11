# src/moshousapient/jobs/queue_inference_job.py
"""
一個獨立的、一次性的背景作業，用於處理佇列中的 'file_inference' 任務。

由 Scheduler 根據系統負載動態啟動。

生命週期:
1. 啟動後，從任務佇列預留一個 'file_inference' 任務。
2. 如果沒有可用任務，則直接退出。
3. 對任務中的影片檔案執行完整的 AI 推論流程。
4. 將推論結果交由 FileEventProducer 處理，由其創建後續的影片編碼任務。
5. 將原始任務標記為 'complete' 或 'failed'。
6. 退出。
"""

# 1. 標準庫導入
import logging
import logging.handlers
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, cast

# 2. 第三方庫導入
import cv2
import numpy as np
import yaml
from shapely.geometry import Point
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT

# 確保在作為獨立腳本執行時能找到 moshousapient 模組
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 3. 本專案導入
from moshousapient.configs.behavior_config import Config as BehaviorConfigMain
from moshousapient.configs.settings_config import settings
from moshousapient.processors.file_event_producer import FileEventProducer
from moshousapient.services.task_queue_service import TaskQueueService
from moshousapient.utils.behavior_analysis_utils import (
    analyze_roi_status,
    analyze_tripwire_crossings,
)
from moshousapient.utils.geometry_utils import calculate_anchor_points


def setup_logging_for_job():
    """為此獨立 Job 設定日誌記錄。"""
    log_dir = settings.DATA_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / "queue_inference_job.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - PID:%(process)d - %(levelname)-8s - %(message)s",
        handlers=[
            logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_models() -> Dict[str, Any]:
    """載入並預熱所有需要的 AI 模型。"""
    try:
        logging.debug(f"正在從 {settings.MODEL_PATH} 載入 TensorRT 模型...")
        model = YOLO(settings.MODEL_PATH, task="detect")

        logging.debug(f"正在載入 {settings.REID_MODEL_PATH} 作為特徵提取器...")
        reid_model = YOLO(settings.REID_MODEL_PATH)

        warmup_frame = np.zeros((settings.ANALYSIS_HEIGHT, settings.ANALYSIS_WIDTH, 3), dtype=np.uint8)
        model.predict(warmup_frame, device=0, verbose=False)
        reid_model.embed(warmup_frame, device=0, verbose=False)
        logging.info("AI 偵測與 Re-ID 模型已成功載入並預熱。")

        logging.debug("正在初始化追蹤器...")
        with open(settings.TRACKER_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        tracker_args = SimpleNamespace(**cfg_dict)
        tracker_args.with_reid = True
        tracker = BOTSORT(args=tracker_args)
        logging.info("BoT-SORT 追蹤器已成功初始化。")

    except Exception:
        logging.critical("載入 AI 模型時發生嚴重錯誤", exc_info=True)
        return {}
    else:
        return {"detector": model, "reid": reid_model, "tracker": tracker}


def _process_frame_tracks(
    tracks: np.ndarray,
    frame_low_res: np.ndarray,
    reid_model: YOLO,
    frame_count: int,
    reid_interval: int,
    track_last_positions: Dict[Tuple[int, int], Point],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], Point]]:
    """處理單一幀中的追蹤物件，執行 Re-ID 和行為分析。"""
    current_frame_tracks_data = []
    reid_features_map = {}

    if frame_count % reid_interval == 0:
        person_crops, valid_track_ids = [], []
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            crop = frame_low_res[y1:y2, x1:x2]
            if crop.size > 0:
                person_crops.append(crop)
                valid_track_ids.append(int(track[4]))
        if person_crops:
            embeddings = reid_model.embed(person_crops, verbose=False)
            for i, track_id in enumerate(valid_track_ids):
                reid_features_map[track_id] = embeddings[i].cpu().numpy().tolist()

    roi_status = analyze_roi_status(
        tracks=tracks,
        roi_enabled=BehaviorConfigMain.ROI_ENABLED,
        roi_polygon=BehaviorConfigMain.ROI_POLYGON_OBJECT,
        roi_settings=BehaviorConfigMain.ROI_SETTINGS,
        global_anchor_points=BehaviorConfigMain.ANCHOR_POINTS,
    )
    crossed_status, updated_last_positions = analyze_tripwire_crossings(
        tracks=tracks,
        track_last_positions=track_last_positions,
        tripwires_enabled=BehaviorConfigMain.TRIPWIRES_ENABLED,
        tripwire_line_objects=BehaviorConfigMain.TRIPWIRE_LINE_OBJECTS,
        global_anchor_points=BehaviorConfigMain.ANCHOR_POINTS,
    )

    for track in tracks:
        track_id = int(track[4])
        bbox = track[:4]

        vis_anchor_strategy = BehaviorConfigMain.ANCHOR_POINTS
        if track_id in crossed_status:
            for tripwire_obj in BehaviorConfigMain.TRIPWIRE_LINE_OBJECTS:
                vis_anchor_strategy = tripwire_obj["config"].get("anchor_points", BehaviorConfigMain.ANCHOR_POINTS)
                break
        elif roi_status.get(track_id, False):
            vis_anchor_strategy = BehaviorConfigMain.ROI_SETTINGS.get("anchor_points", BehaviorConfigMain.ANCHOR_POINTS)

        bbox_tuple = cast(Tuple[float, float, float, float], tuple(bbox))
        vis_anchors = calculate_anchor_points(bbox_tuple, vis_anchor_strategy)
        anchor_coords = [next(iter(a.coords)) for a in vis_anchors if isinstance(a, Point)]

        current_frame_tracks_data.append(
            {
                "track_id": track_id,
                "box_xyxy": [float(c) for c in bbox],
                "confidence": float(track[5]) if len(track) > 5 else None,
                "feature": reid_features_map.get(track_id),
                "is_in_roi": roi_status.get(track_id, False),
                "has_crossed_tripwire": track_id in crossed_status,
                "anchors": anchor_coords,
            }
        )
    return current_frame_tracks_data, updated_last_positions


def run_full_inference_on_file(video_path: Path, models: Dict[str, Any]) -> Dict[str, Any]:
    """對指定的影片檔案執行完整的 AI 推論流程。"""
    logging.info(f"開始對影片進行完整 AI 分析: {video_path.name}")
    start_time = time.time()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError("Could not open video file")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    all_frame_data: List[Dict[str, Any]] = []
    track_last_positions: Dict[Tuple[int, int], Point] = {}
    frame_count, reid_interval = 0, 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_low_res = cv2.resize(frame, (settings.ANALYSIS_WIDTH, settings.ANALYSIS_HEIGHT))

        dets_results = models["detector"](frame_low_res, device=0, verbose=False, classes=[0], conf=0.4)
        tracks = models["tracker"].update(dets_results[0].boxes.cpu().numpy(), frame_low_res)

        current_frame_tracks_data = []
        if tracks.size > 0:
            current_frame_tracks_data, track_last_positions = _process_frame_tracks(
                tracks, frame_low_res, models["reid"], frame_count, reid_interval, track_last_positions
            )

        all_frame_data.append({"frame_index": frame_count, "tracks": current_frame_tracks_data})

    cap.release()
    processing_duration = time.time() - start_time
    logging.info(f"影片分析完成。共處理 {frame_count} 幀，耗時 {processing_duration:.2f} 秒。")

    return {
        "video_path": str(video_path),
        "status": "success",
        "analytics": {
            "total_frames": frame_count,
            "source_fps": source_fps,
            "processing_duration_sec": processing_duration,
            "source_width": source_width,
            "source_height": source_height,
        },
        "frames": all_frame_data,
    }


def _validate_and_get_models() -> Dict[str, Any]:
    """載入並驗證 AI 模型。"""
    models = load_models()
    if not models:
        raise RuntimeError("AI model loading failed")
    return models


def _validate_task_payload(payload: Dict[str, Any]) -> Path:
    """驗證任務 payload 的有效性，並在無效時引發異常。"""
    video_path_str = payload.get("video_path")
    video_path = Path(video_path_str) if video_path_str else None
    if not video_path or not video_path.exists():
        raise FileNotFoundError("Task video file not found")
    return video_path


def main():
    """Job 的主入口點。"""
    setup_logging_for_job()
    worker_id = f"queue-job-{os.getpid()}"
    logging.info(f"'{worker_id}' 啟動。")

    task_queue = TaskQueueService()
    task = None
    try:
        task = task_queue.reserve_task(worker_id)
        if not task:
            logging.info("佇列中沒有待處理任務，Job 正常退出。")
            return

        payload = pickle.loads(task["payload"])
        if payload.get("task_type") != "file_inference":
            logging.warning(f"獲取到非預期任務類型 '{payload.get('task_type')}'，將其釋放回佇列。")
            task_queue.fail_task(task["id"], "Requeued for wrong type", requeue=True)
            return

        logging.info(f"已預留任務 ID: {task['id']}，處理檔案: {Path(payload.get('video_path', '')).name}")

        video_path = _validate_task_payload(payload)
        models = _validate_and_get_models()

        BehaviorConfigMain.initialize_static_settings()

        inference_results = run_full_inference_on_file(video_path, models)

        producer = FileEventProducer(notifier=None)
        producer.process_results(inference_results)

        task_queue.complete_task(task["id"])
        logging.info(f"任務 ID: {task['id']} 已成功處理並完成。")

    except Exception as e:
        logging.error(f"處理任務時發生未預期錯誤: {e}", exc_info=True)
        if task:
            task_queue.fail_task(task["id"], f"Unhandled exception: {e}")
    finally:
        task_queue.close_connection()
        logging.info(f"'{worker_id}' 退出。")


if __name__ == "__main__":
    main()
