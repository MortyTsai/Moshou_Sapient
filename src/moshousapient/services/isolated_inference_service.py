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

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon, LineString, Point
from shapely.errors import ShapelyError
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT

try:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from moshousapient.settings import settings
    from moshousapient.utils.geometry_utils import get_point_side_of_line
except ImportError as e:
    print(f"緊急錯誤: 無法導入 MoshouSapient 核心模組。請確保從專案根目錄執行。錯誤: {e}", file=sys.stderr)
    sys.exit(1)

# --- 全域設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [IsolatedInferenceService] - %(levelname)s - %(message)s',
                    stream=sys.stdout)


class BehaviorConfig:
    """在隔離服務中載入並管理行為分析規則"""
    ROI_ENABLED: bool = False
    ROI_POLYGON_OBJECT: Union[Polygon, None] = None
    ROI_DWELL_TIME_THRESHOLD: float = 3.0
    TRIPWIRES_ENABLED: bool = False
    TRIPWIRE_LINE_OBJECTS: List[Dict[str, Any]] = []

    @staticmethod
    def load_from_yaml(config_path: Path):
        if not config_path.exists():
            logging.warning(f"行為分析設定檔不存在: {config_path}。將停用高階行為分析。")
            return
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}

            # 載入 ROI 設定
            roi_settings = config_data.get('roi', {})
            if roi_settings and roi_settings.get('enabled', False):
                polygon_points = roi_settings.get('polygon_points', [])
                if polygon_points and len(polygon_points) >= 3:
                    BehaviorConfig.ROI_POLYGON_OBJECT = Polygon(polygon_points)
                    BehaviorConfig.ROI_ENABLED = True
                    BehaviorConfig.ROI_DWELL_TIME_THRESHOLD = roi_settings.get('dwell_time_threshold', 3.0)
                    logging.info(f"成功載入 ROI 區域，面積: {BehaviorConfig.ROI_POLYGON_OBJECT.area:.2f} 平方像素。")
                else:
                    logging.warning("ROI 已啟用但未提供有效的多邊形座標點 (至少3個點)。ROI 功能已停用。")

            # 載入 Tripwire 設定
            tripwire_settings = config_data.get('tripwires', {})
            if tripwire_settings and tripwire_settings.get('enabled', False):
                lines = tripwire_settings.get('lines', [])
                BehaviorConfig.TRIPWIRE_LINE_OBJECTS.clear()
                for line_config in lines:
                    points = line_config.get("points")
                    if points and len(points) == 2:
                        line = LineString(points)
                        direction = line_config.get("alert_direction", "both")
                        BehaviorConfig.TRIPWIRE_LINE_OBJECTS.append({"line": line, "direction": direction})
                if BehaviorConfig.TRIPWIRE_LINE_OBJECTS:
                    BehaviorConfig.TRIPWIRES_ENABLED = True
                    logging.info(f"成功載入 {len(BehaviorConfig.TRIPWIRE_LINE_OBJECTS)} 條警戒線。")
        except (yaml.YAMLError, ShapelyError, TypeError) as e:
            logging.error(f"解析行為分析設定檔時發生錯誤: {e}。將停用高階行為分析。")


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
        reid_model.predict(warmup_frame, device=0, verbose=False)
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
        tracks = tracker.update(dets_results[0].boxes.cpu(), frame_low_res)

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
                x1, y1, x2, y2 = track[:4]

                is_in_roi = False
                if BehaviorConfig.ROI_ENABLED and BehaviorConfig.ROI_POLYGON_OBJECT:
                    bottom_center_point = Point((x1 + x2) / 2, y2)
                    is_in_roi = BehaviorConfig.ROI_POLYGON_OBJECT.contains(bottom_center_point)

                has_crossed_tripwire = False
                current_position = Point((x1 + x2) / 2, y2)
                last_position = track_last_positions.get(track_id)

                if BehaviorConfig.TRIPWIRES_ENABLED and last_position and last_position != current_position:
                    movement_line = LineString([last_position, current_position])
                    for tripwire_obj in BehaviorConfig.TRIPWIRE_LINE_OBJECTS:
                        tripwire_line, alert_direction = tripwire_obj["line"], tripwire_obj["direction"]
                        if movement_line.intersects(tripwire_line):
                            p1, p2 = tripwire_line.coords
                            side_before = get_point_side_of_line(last_position, Point(p1), Point(p2))
                            side_after = get_point_side_of_line(current_position, Point(p1), Point(p2))
                            if side_before != 0 and side_after != 0 and side_before != side_after:
                                crossed_to_right = side_before == 1 and side_after == -1
                                crossed_to_left = side_before == -1 and side_after == 1
                                if (alert_direction == "both" or
                                        (alert_direction == "cross_to_right" and crossed_to_right) or
                                        (alert_direction == "cross_to_left" and crossed_to_left)):
                                    has_crossed_tripwire = True
                                    break

                track_last_positions[track_id] = current_position

                current_frame_tracks.append({
                    "track_id": track_id, "box_xyxy": [float(coord) for coord in track[:4]],
                    "confidence": float(track[5]), "feature": reid_features_map.get(track_id),
                    "is_in_roi": is_in_roi, "has_crossed_tripwire": has_crossed_tripwire
                })

            disappeared_ids = set(track_last_positions.keys()) - current_tracked_ids
            for track_id in disappeared_ids:
                del track_last_positions[track_id]

        all_frame_data.append({"frame_index": frame_count, "tracks": current_frame_tracks})

    cap.release()
    end_time = time.time()
    processing_duration = end_time - start_time
    logging.info(f"影片分析完成。共處理 {frame_count} 幀，耗時 {processing_duration:.2f} 秒。")

    final_results = {
        "video_path": str(video_path), "status": "success",
        "analytics": {
            "total_frames": frame_count, "source_fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "processing_duration_sec": processing_duration,
        },
        "frames": all_frame_data
    }

    logging.info(f"正在將追蹤資料寫入 JSON 檔案: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f)


def main():
    """主函式：解析參數並啟動推論"""
    parser = argparse.ArgumentParser(description="MoshouSapient - 獨立 AI 推論服務")
    parser.add_argument('--video-path', type=Path, required=True)
    parser.add_argument('--output-json-path', type=Path, required=True)
    parser.add_argument('--behavior-config-path', type=Path, required=True)
    args = parser.parse_args()

    if not settings:
        logging.error("設定模組未成功載入。")
        sys.exit(1)

    BehaviorConfig.load_from_yaml(args.behavior_config_path)

    models = load_models()
    if not models:
        sys.exit(1)

    run_inference(args.video_path, args.output_json_path, models)


if __name__ == "__main__":
    main()