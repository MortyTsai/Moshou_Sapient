# src/moshousapient/services/isolated_video_processor.py (Definitive Final Cleaned Version 2)
import argparse, logging, sys, subprocess, pickle, os
from pathlib import Path
import cv2
import numpy as np
from moshousapient.config import Config

# 將導入和初始化移到全域，避免 except 作用域問題
try:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


    # 預先初始化，這樣即使檔案不存在，Config 類也已載入
    Config.initialize_static_settings()
except ImportError as e:
    print(f"緊急錯誤: 無法導入 MoshouSapient 核心模組。錯誤: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [VideoProcessorService] - %(levelname)s - %(message)s')


def process_video_event(input_data_path: str, output_path: str, event_type: str):
    logging.info(f"開始處理事件 '{event_type}'，從資料檔案: {input_data_path}")

    frames_data = []
    try:
        with open(input_data_path, 'rb') as f:
            frames_data = pickle.load(f)
    except Exception as e:
        logging.error(f"無法讀取或反序列化輸入資料檔案: {e}")
    finally:
        # 修正：移除對 'e' 的隱藏引用
        if os.path.exists(input_data_path):
            try:
                os.remove(input_data_path)
                logging.info(f"已清理臨時資料檔案: {input_data_path}")
            except OSError as remove_error:
                logging.error(f"清理臨時資料檔案失敗: {remove_error}")

    if not frames_data:
        logging.warning("輸入資料為空，不進行處理。")
        return

    start_time = frames_data[0]['time']
    end_time = frames_data[-1]['time']
    duration = end_time - start_time
    actual_fps = len(frames_data) / duration if duration > 0 else Config.TARGET_FPS
    height, width, _ = frames_data[0]['frame'].shape

    ffmpeg_encode_cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24', '-r', str(actual_fps), '-i', '-',
        '-c:v', 'hevc_nvenc', '-preset', 'p6', '-an', output_path
    ]

    process = subprocess.Popen(ffmpeg_encode_cmd, stdin=subprocess.PIPE)
    logging.info(f"開始對 {len(frames_data)} 幀影像進行繪圖和硬體編碼...")

    for frame_data in frames_data:
        frame = frame_data['frame']
        overlay = frame.copy()

        # 修正：使用正確的屬性名 ROI_POLYGON_OBJECT
        if Config.ROI_ENABLED and Config.ROI_POLYGON_OBJECT:
            roi_points = np.array(Config.ROI_POLYGON_OBJECT.exterior.coords, dtype=np.int32)
            cv2.fillPoly(overlay, [roi_points], color=(255, 255, 0))
            cv2.polylines(overlay, [roi_points], isClosed=True, color=(255, 255, 0), thickness=2)

        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        current_frame_alert_ids = frame_data.get('tripwire_alert_ids', set())

        for track in frame_data.get('tracks', []):
            box, track_id = track[:4], int(track[4])
            x1, y1, x2, y2 = map(int, box)
            track_roi_status = frame_data.get('track_roi_status', {})

            box_color = (0, 255, 0)
            if track_roi_status.get(track_id, False):
                box_color = (0, 255, 255)
            if track_id in current_frame_alert_ids:
                box_color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        if process.stdin:
            process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()
    logging.info(f"事件影片已成功處理並儲存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MoshouSapient - 獨立影片處理服務")
    parser.add_argument("--input-data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--event-type", required=True)
    args = parser.parse_args()
    process_video_event(args.input_data_path, args.output_path, args.event_type)


if __name__ == "__main__":
    main()