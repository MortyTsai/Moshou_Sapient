# src/moshousapient/services/isolated_inference_service.py

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any
from types import SimpleNamespace

settings = None
try:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from moshousapient.settings import settings
except ImportError as e:
    print(f"緊急錯誤: 無法導入 MoshouSapient 設定模組。 {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [IsolatedInferenceService] - %(levelname)s - %(message)s',
                    stream=sys.stdout)


def load_models() -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            logging.error("嚴重錯誤: 在隔離服務中未偵測到 CUDA 設備。")
            return {}

        logging.info(f"偵測到 GPU: {torch.cuda.get_device_name(0)}")

        model_path = settings.MODEL_PATH
        logging.info(f"正在從 {model_path} 載入 TensorRT 偵測模型...")
        model = YOLO(model_path, task='detect')

        reid_model_path = settings.REID_MODEL_PATH
        logging.info(f"正在從 {reid_model_path} 載入 Re-ID 模型...")
        reid_model = YOLO(reid_model_path)

        logging.info("正在預熱 AI 模型...")
        warmup_frame = np.zeros((settings.ANALYSIS_HEIGHT, settings.ANALYSIS_WIDTH, 3), dtype=np.uint8)
        model.predict(warmup_frame, device=0, verbose=False, classes=[0])
        reid_model.predict(warmup_frame, device=0, verbose=False)
        logging.info("AI 模型已成功載入並預熱。")

        return {"detector": model, "reid": reid_model}
    except Exception as model_load_exc:
        logging.error(f"載入 AI 模型時發生嚴重錯誤: {model_load_exc}", exc_info=True)
        return {}


def initialize_tracker() -> Any:
    try:
        import yaml
        from ultralytics.trackers import BOTSORT

        tracker_config_path = settings.TRACKER_CONFIG_PATH
        logging.info(f"正在從 {tracker_config_path} 解析追蹤器設定...")
        with open(tracker_config_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

        tracker_args = SimpleNamespace(**cfg_dict)
        tracker_args.with_reid = True

        tracker = BOTSORT(args=tracker_args)
        logging.info("追蹤器 (BoT-SORT, with Re-ID) 已成功初始化。")
        return tracker
    except Exception as tracker_init_exc:
        logging.error(f"初始化追蹤器時發生錯誤: {tracker_init_exc}", exc_info=True)
        return None


def run_inference(video_path: Path, output_json_path: Path, models: Dict[str, Any]) -> None:
    logging.info(f"開始處理影片: {video_path}")
    start_time = time.time()

    try:
        import cv2
        import numpy as np
    except ImportError:
        logging.error("OpenCV 或 NumPy 未安裝，無法處理影片。")
        sys.exit(1)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame_low_res = cv2.resize(frame, (settings.ANALYSIS_WIDTH, settings.ANALYSIS_HEIGHT))

        dets_results = detector(frame_low_res, device=0, verbose=False, classes=[0], conf=0.4)
        boxes_on_cpu = dets_results[0].boxes.cpu()
        tracks = tracker.update(boxes_on_cpu, frame_low_res)

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

            for track in tracks:
                track_id = int(track[4])
                current_frame_tracks.append({
                    "track_id": track_id,
                    "box_xyxy": [float(coord) for coord in track[:4]],
                    "confidence": float(track[5]),
                    "feature": reid_features_map.get(track_id)  # 如果當前幀提取了特徵，則加入
                })

        all_frame_data.append({
            "frame_index": frame_count,
            "tracks": current_frame_tracks
        })

    cap.release()
    end_time = time.time()
    processing_duration = end_time - start_time
    logging.info(f"影片分析完成。共處理 {frame_count} 幀，耗時 {processing_duration:.2f} 秒。")

    final_results = {
        "video_path": str(video_path),
        "status": "success",
        "analytics": {
            "total_frames": frame_count,
            "source_fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "processing_duration_sec": processing_duration,
        },
        "frames": all_frame_data
    }

    logging.info(f"正在將追蹤資料寫入 JSON 檔案: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f)


def main():
    parser = argparse.ArgumentParser(description="MoshouSapient - 獨立 AI 推論服務")
    parser.add_argument('--video-path', type=Path, required=True)
    parser.add_argument('--output-json-path', type=Path, required=True)
    args = parser.parse_args()

    if not settings: sys.exit(1)
    models = load_models()
    if not models: sys.exit(1)
    run_inference(args.video_path, args.output_json_path, models)


if __name__ == "__main__":
    main()