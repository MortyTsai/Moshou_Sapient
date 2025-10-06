# src/moshousapient/services/isolated_inference_service.py
"""
一個獨立的 AI 推論服務腳本。
此腳本被設計為透過 subprocess 呼叫，以實現程序級隔離。
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Union, List, cast, Tuple

import cv2
import numpy as np
import torch
import yaml
from shapely.errors import ShapelyError
from shapely.geometry import Polygon, LineString, Point
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT

# 確保在作為獨立腳本執行時能找到 moshousapient 模組
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from moshousapient.configs.settings_config import settings
from moshousapient.utils.geometry_utils import calculate_anchor_points
from moshousapient.utils.behavior_analysis_utils import analyze_roi_status, analyze_tripwire_crossings


class BehaviorConfig:
    """在隔離服務中載入並管理行為分析規則。"""
    ANCHOR_POINTS: Union[str, List[str]] = 'bottom_center'
    ROI_ENABLED: bool = False
    ROI_SETTINGS: Dict[str, Any] = {}
    ROI_POLYGON_OBJECT: Union[Polygon, None] = None
    TRIPWIRES_ENABLED: bool = False
    TRIPWIRE_SETTINGS: Dict[str, Any] = {}
    TRIPWIRE_LINE_OBJECTS: List[Dict[str, Any]] = []

    @staticmethod
    def load_from_yaml(config_path: Path):
        """從指定的 YAML 檔案載入並解析所有行為分析規則。"""
        if not config_path.exists():
            logging.warning(f"行為分析設定檔不存在: {config_path}。將停用高階分析。")
            return
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}

            BehaviorConfig.ANCHOR_POINTS = config_data.get('anchor_points', 'bottom_center')
            BehaviorConfig.ROI_SETTINGS = config_data.get('roi', {})
            if BehaviorConfig.ROI_SETTINGS.get('enabled', False):
                polygon_points = BehaviorConfig.ROI_SETTINGS.get('polygon_points', [])
                if polygon_points and len(polygon_points) >= 3:
                    BehaviorConfig.ROI_POLYGON_OBJECT = Polygon(polygon_points)
                    BehaviorConfig.ROI_ENABLED = True
                    logging.info(f"成功載入 ROI 區域，面積: {BehaviorConfig.ROI_POLYGON_OBJECT.area:.2f}px。")

            BehaviorConfig.TRIPWIRE_SETTINGS = config_data.get('tripwires', {})
            if BehaviorConfig.TRIPWIRE_SETTINGS.get('enabled', False):
                lines = BehaviorConfig.TRIPWIRE_SETTINGS.get('lines', [])
                BehaviorConfig.TRIPWIRE_LINE_OBJECTS.clear()
                for line_config in lines:
                    points = line_config.get("points")
                    if points and len(points) == 2:
                        line = LineString(points)
                        direction = line_config.get("alert_direction", "both")
                        BehaviorConfig.TRIPWIRE_LINE_OBJECTS.append(
                            {'line': line, "direction": direction, "config": line_config}
                        )
                if BehaviorConfig.TRIPWIRE_LINE_OBJECTS:
                    BehaviorConfig.TRIPWIRES_ENABLED = True
                    logging.info(f"成功載入 {len(BehaviorConfig.TRIPWIRE_LINE_OBJECTS)} 條警戒線。")
        except (yaml.YAMLError, ShapelyError, TypeError) as e:
            logging.error(f"解析行為分析設定檔時發生錯誤: {e}。")


def load_models() -> Dict[str, Any]:
    """載入並預熱偵測與 Re-ID 模型。"""
    try:
        if not torch.cuda.is_available():
            logging.error("嚴重錯誤: 未偵測到 CUDA 設備。")
            return {}
        logging.info(f"偵測到 GPU: {torch.cuda.get_device_name(0)}")
        logging.info("正在預熱 AI 模型...")
        model = YOLO(settings.MODEL_PATH, task='detect')
        reid_model = YOLO(settings.REID_MODEL_PATH)
        warmup_frame = np.zeros((settings.ANALYSIS_HEIGHT, settings.ANALYSIS_WIDTH, 3), dtype=np.uint8)
        model.predict(warmup_frame, device=0, verbose=False, classes=[0])
        reid_model.embed(warmup_frame, device=0, verbose=False)
        logging.info("AI 模型已成功載入並預熱。")
        return {"detector": model, "reid": reid_model}
    except Exception as e:
        logging.critical(f"載入 AI 模型時發生嚴重錯誤: {e}", exc_info=True)
        return {}


def initialize_tracker() -> Any:
    """根據設定檔初始化追蹤器物件。"""
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
    """對指定的影片檔案執行完整的 AI 推論流程。"""
    logging.info(f"開始處理影片: {video_path.name}")
    start_time = time.time()
    detector = models.get("detector")
    reid_model = models.get("reid")
    tracker = initialize_tracker()
    if not all([detector, reid_model, tracker]):
        logging.critical("一個或多個 AI 模型未能成功初始化。推論終止。")
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"錯誤: 無法開啟影片檔案 {video_path}")
        return
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    all_frame_data, track_last_positions, frame_count, reid_interval = [], {}, 0, 5
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        frame_low_res = cv2.resize(frame, (settings.ANALYSIS_WIDTH, settings.ANALYSIS_HEIGHT))
        dets_results = detector(frame_low_res, device=0, verbose=False, classes=[0], conf=0.4)
        tracks = tracker.update(dets_results[0].boxes.cpu().numpy(), frame_low_res)
        current_frame_tracks_data = []
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
            roi_status = analyze_roi_status(tracks=tracks, roi_enabled=BehaviorConfig.ROI_ENABLED,
                                            roi_polygon=BehaviorConfig.ROI_POLYGON_OBJECT,
                                            roi_settings=BehaviorConfig.ROI_SETTINGS,
                                            global_anchor_points=BehaviorConfig.ANCHOR_POINTS)
            crossed_status, track_last_positions = analyze_tripwire_crossings(tracks=tracks,
                                                                              track_last_positions=track_last_positions,
                                                                              tripwires_enabled=BehaviorConfig.TRIPWIRES_ENABLED,
                                                                              tripwire_line_objects=BehaviorConfig.TRIPWIRE_LINE_OBJECTS,
                                                                              global_anchor_points=BehaviorConfig.ANCHOR_POINTS)
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                vis_anchor_strategy = BehaviorConfig.ANCHOR_POINTS
                if track_id in crossed_status:
                    for tripwire_obj in BehaviorConfig.TRIPWIRE_LINE_OBJECTS:
                        vis_anchor_strategy = tripwire_obj["config"].get('anchor_points', BehaviorConfig.ANCHOR_POINTS)
                        break
                elif roi_status.get(track_id, False):
                    vis_anchor_strategy = BehaviorConfig.ROI_SETTINGS.get('anchor_points', BehaviorConfig.ANCHOR_POINTS)
                bbox_tuple = cast(Tuple[float, float, float, float], tuple(bbox))
                vis_anchors = calculate_anchor_points(bbox_tuple, vis_anchor_strategy)
                anchor_coords = [list(a.coords)[0] for a in vis_anchors if isinstance(a, Point)]
                current_frame_tracks_data.append({"track_id": track_id, "box_xyxy": [float(c) for c in bbox],
                                                  "confidence": float(track[5]) if len(track) > 5 else None,
                                                  "feature": reid_features_map.get(track_id),
                                                  "is_in_roi": roi_status.get(track_id, False),
                                                  "has_crossed_tripwire": track_id in crossed_status,
                                                  "anchors": anchor_coords})
        all_frame_data.append({"frame_index": frame_count, "tracks": current_frame_tracks_data})
    cap.release()
    processing_duration = time.time() - start_time
    logging.info(f"影片分析完成。共處理 {frame_count} 幀，耗時 {processing_duration:.2f} 秒。")
    final_results = {"video_path": str(video_path), "status": "success",
                     "analytics": {"total_frames": frame_count, "source_fps": source_fps,
                                   "processing_duration_sec": processing_duration, "source_width": source_width,
                                   "source_height": source_height}, "frames": all_frame_data}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f)
    logging.info(f"追蹤資料已寫入 JSON 檔案: {output_json_path}")


def main():
    """腳本主入口點。"""
    parser = argparse.ArgumentParser(description="MoshouSapient - 獨立 AI 推論服務")
    parser.add_argument('--video-path', type=Path, required=True)
    parser.add_argument('--output-json-path', type=Path, required=True)
    parser.add_argument('--behavior-config-path', type=Path, required=True)
    args = parser.parse_args()

    # 為此獨立腳本配置一個簡單的日誌記錄器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - [IsoInference] %(message)s',
        datefmt='%H:%M:%S'
    )

    BehaviorConfig.load_from_yaml(args.behavior_config_path)
    models = load_models()
    if not models:
        sys.exit(1)

    run_inference(args.video_path, args.output_json_path, models)


if __name__ == "__main__":
    main()