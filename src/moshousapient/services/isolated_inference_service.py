# src/moshousapient/services/isolated_inference_service.py
import argparse
import json
import logging
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Union, List
from types import SimpleNamespace
from moshousapient.settings import settings
from moshousapient.utils.geometry_utils import calculate_anchor_points, get_point_side_of_line

try:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
except ImportError as import_err:
    print(f"緊急錯誤: 無法導入 MoshouSapient 核心模組。錯誤: {import_err}", file=sys.stderr)
    sys.exit(1)

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon, LineString, Point
from shapely.errors import ShapelyError
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT


class BehaviorConfig:
    """在隔離服務中載入並管理行為分析規則"""
    ANCHOR_POINTS: Union[str, List[str]] = 'bottom_center'
    ROI_ENABLED: bool = False
    ROI_POLYGON_OBJECT: Union[Polygon, None] = None
    ROI_DWELL_TIME_THRESHOLD: float = 3.0
    TRIPWIRES_ENABLED: bool = False
    TRIPWIRE_LINE_OBJECTS: List[Dict[str, Any]] = []

    @staticmethod
    def load_from_yaml(config_path: Path):
        if not config_path.exists():
            logging.warning(f"行為分析設定檔不存在: {config_path}。")
            return
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}

            BehaviorConfig.ANCHOR_POINTS = config_data.get('anchor_points', 'bottom_center')
            roi_settings = config_data.get('roi', {})
            if roi_settings and roi_settings.get('enabled', False):
                polygon_points = roi_settings.get('polygon_points', [])
                if polygon_points and len(polygon_points) >= 3:
                    BehaviorConfig.ROI_POLYGON_OBJECT = Polygon(polygon_points)
                    BehaviorConfig.ROI_ENABLED = True
                    BehaviorConfig.ROI_DWELL_TIME_THRESHOLD = roi_settings.get('dwell_time_threshold', 3.0)
                    logging.info(f"成功載入 ROI 區域, 面積: {BehaviorConfig.ROI_POLYGON_OBJECT.area:.2f}px。")

            tripwire_settings = config_data.get('tripwires', {})
            if tripwire_settings and tripwire_settings.get('enabled', False):
                lines = tripwire_settings.get('lines', [])
                BehaviorConfig.TRIPWIRE_LINE_OBJECTS.clear()
                for line_config in lines:
                    points = line_config.get("points")
                    if points and len(points) == 2:
                        line = LineString(points)
                        direction = line_config.get("alert_direction", "both")
                        BehaviorConfig.TRIPWIRE_LINE_OBJECTS.append({'line': line, "direction": direction})
                if BehaviorConfig.TRIPWIRE_LINE_OBJECTS:
                    BehaviorConfig.TRIPWIRES_ENABLED = True
                    logging.info(f"成功載入 {len(BehaviorConfig.TRIPWIRE_LINE_OBJECTS)} 條警戒線。")
        except (yaml.YAMLError, ShapelyError, TypeError) as e:
            logging.error(f"解析行為分析設定檔時發生錯誤: {e}。")


def load_models() -> Dict[str, Any]:
    """載入並預熱偵測與 Re-ID 模型"""
    try:
        if not torch.cuda.is_available():
            logging.error("嚴重錯誤: 未偵測到 CUDA 設備。")
            return {}
        logging.info(f"偵測到 GPU: {torch.cuda.get_device_name(0)}")
        model = YOLO(settings.MODEL_PATH, task='detect')
        reid_model = YOLO(settings.REID_MODEL_PATH)
        logging.info("正在預熱 AI 模型...")
        warmup_frame = np.zeros((settings.ANALYSIS_HEIGHT, settings.ANALYSIS_WIDTH, 3), dtype=np.uint8)
        model.predict(warmup_frame, device=0, verbose=False, classes=[0])
        reid_model.embed(warmup_frame, device=0, verbose=False)
        logging.info("AI 模型已成功載入並預熱。")
        return {"detector": model, "reid": reid_model}
    except Exception as e:
        logging.error(f"載入 AI 模型時發生嚴重錯誤: {e}", exc_info=True)
        return {}


def initialize_tracker() -> Any:
    """根據設定檔初始化追蹤器"""
    try:
        with open(settings.TRACKER_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        tracker_args = SimpleNamespace(**cfg_dict)
        tracker_args.with_reid = True
        tracker = BOTSORT(args=tracker_args)
        logging.info("追蹤器 (BoT-SORT, with Re-ID) 已成功初始化。")
        return tracker
    except Exception as e:
        logging.error(f"初始化追蹤器時發生錯誤: {e}", exc_info=True)
        return None


def run_inference(video_path: Path, output_json_path: Path, models: Dict[str, Any]):
    """對指定的影片檔案執行完整的 AI 推論流程"""
    logging.info(f"開始處理影片: {video_path}")
    start_time = time.time()

    detector = models.get("detector")
    reid_model = models.get("reid")
    tracker = initialize_tracker()

    if not all([detector, reid_model, tracker]):
        logging.error("一個或多個 AI 模型未能成功初始化。")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"錯誤: 無法開啟影片檔案 {video_path}")
        sys.exit(1)

    frame_count = 0
    reid_interval = 5
    all_frame_data = []
    track_last_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame_low_res = cv2.resize(frame, (settings.ANALYSIS_WIDTH, settings.ANALYSIS_HEIGHT))
        dets_results = detector(frame_low_res, device=0, verbose=False, classes=[0], conf=0.4)
        tracks = tracker.update(dets_results[0].boxes.cpu().numpy(), frame_low_res)

        current_frame_tracks = []
        if tracks.size > 0:
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

            current_tracked_ids = set()
            for track in tracks:
                track_id = int(track[4])
                current_tracked_ids.add(track_id)
                bbox = track[:4]

                has_crossed_tripwire = False
                if BehaviorConfig.TRIPWIRES_ENABLED:
                    anchor_strategy = BehaviorConfig.ANCHOR_POINTS
                    current_anchors = calculate_anchor_points(bbox, anchor_strategy)

                    for i, current_anchor in enumerate(current_anchors):
                        if not isinstance(current_anchor, Point): continue

                        anchor_key = (track_id, i)
                        last_position = track_last_positions.get(anchor_key)

                        if last_position and last_position != current_anchor:
                            movement_line = LineString([last_position, current_anchor])
                            for tripwire_obj in BehaviorConfig.TRIPWIRE_LINE_OBJECTS:
                                tripwire_line = tripwire_obj["line"]
                                alert_direction = tripwire_obj["direction"]

                                if movement_line.intersects(tripwire_line):
                                    p1_coords, p2_coords = tripwire_line.coords[0], tripwire_line.coords[1]
                                    p1, p2 = Point(p1_coords), Point(p2_coords)

                                    side_before = get_point_side_of_line(last_position, p1, p2)
                                    side_after = get_point_side_of_line(current_anchor, p1, p2)

                                    if side_before != 0 and side_after != 0 and side_before != side_after:
                                        # --- 修正開始：反轉左右判斷以適應螢幕座標系 ---
                                        crossed_to_right = side_before == -1 and side_after == 1
                                        crossed_to_left = side_before == 1 and side_after == -1
                                        # --- 修正結束 ---

                                        should_alert = (alert_direction == "both" or
                                                        (alert_direction == "cross_to_right" and crossed_to_right) or
                                                        (alert_direction == "cross_to_left" and crossed_to_left))

                                        if should_alert:
                                            has_crossed_tripwire = True
                                            break

                        track_last_positions[anchor_key] = current_anchor
                        if has_crossed_tripwire: break

                is_in_roi = False
                if BehaviorConfig.ROI_ENABLED and BehaviorConfig.ROI_POLYGON_OBJECT:
                    anchor_strategy = BehaviorConfig.ANCHOR_POINTS
                    anchors = calculate_anchor_points(bbox, anchor_strategy)
                    for anchor in anchors:
                        if isinstance(anchor, Point) and BehaviorConfig.ROI_POLYGON_OBJECT.contains(anchor):
                            is_in_roi = True
                            break

                anchors = calculate_anchor_points(bbox, BehaviorConfig.ANCHOR_POINTS)
                anchor_coords = [list(a.coords)[0] for a in anchors if isinstance(a, Point)]

                current_frame_tracks.append({
                    "track_id": track_id, "box_xyxy": [float(c) for c in bbox],
                    "confidence": float(track[5]) if len(track) > 5 else None,
                    "feature": reid_features_map.get(track_id), "is_in_roi": is_in_roi,
                    "has_crossed_tripwire": has_crossed_tripwire,
                    "anchors": anchor_coords
                })

            disappeared_anchor_keys = set(track_last_positions.keys())
            for track in tracks:
                track_id = int(track[4])
                num_anchors = len(calculate_anchor_points(track[:4], BehaviorConfig.ANCHOR_POINTS))
                for i in range(num_anchors):
                    disappeared_anchor_keys.discard((track_id, i))

            for key in disappeared_anchor_keys:
                if key in track_last_positions:
                    del track_last_positions[key]

        all_frame_data.append({"frame_index": frame_count, "tracks": current_frame_tracks})

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    end_time = time.time()
    processing_duration = end_time - start_time
    logging.info(f"影片分析完成。共處理 {frame_count} 幀, 耗時 {processing_duration:.2f} 秒。")

    final_results = {
        "video_path": str(video_path), "status": "success",
        "analytics": {"total_frames": frame_count, "source_fps": source_fps,
                      "processing_duration_sec": processing_duration},
        "frames": all_frame_data
    }

    logging.info(f"正在將追蹤資料寫入 JSON 檔案: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f)

def main():
    """主函式：解析參數並啟動推論"""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - [IsolatedInferenceService] - %(levelname)s - %(message)s',
                        stream=sys.stdout)

    parser = argparse.ArgumentParser(description="MoshouSapient - 獨立 AI 推論服務")
    parser.add_argument('--video-path', type=Path, required=True)
    parser.add_argument('--output-json-path', type=Path, required=True)
    parser.add_argument('--behavior-config-path', type=Path, required=True)
    args = parser.parse_args()

    BehaviorConfig.load_from_yaml(args.behavior_config_path)
    models = load_models()
    if not models:
        sys.exit(1)

    run_inference(args.video_path, args.output_json_path, models)


if __name__ == "__main__":
    main()