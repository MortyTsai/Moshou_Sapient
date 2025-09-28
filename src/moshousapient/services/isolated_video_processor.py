# src/moshousapient/services/isolated_video_processor.py (Definitive Final Cleaned Version 2)
import argparse, logging, sys, subprocess, pickle, os
from pathlib import Path
import cv2
import numpy as np
from moshousapient.config import Config

try:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


    Config.initialize_static_settings()
except ImportError as e:
    print(f"緊急錯誤: 無法導入 MoshouSapient 核心模組。錯誤: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [VideoProcessorService] - %(levelname)s - %(message)s')


def process_video_event(input_data_path: str, output_path: str, event_type: str):
    """
    處理單個影片事件：讀取幀資料、繪製視覺化元素、並使用 FFmpeg 硬體編碼儲存影片。
    此函式在一個獨立的子程序中執行。
    """
    logging.info(f"開始處理事件 '{event_type}', 從資料檔案: {input_data_path}")

    frames_data = []
    try:
        with open(input_data_path, 'rb') as f:
            frames_data = pickle.load(f)
    except Exception as e:
        logging.error(f"無法讀取或反序列化輸入資料檔案: {e}")
    finally:
        if os.path.exists(input_data_path):
            try:
                os.remove(input_data_path)
                logging.info(f"已清理臨時資料檔案: {input_data_path}")
            except OSError as remove_error:
                logging.error(f"清理臨時資料檔案失敗: {remove_error}")

    if not frames_data:
        logging.warning("輸入資料為空, 不進行處理。")
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

    for frame_idx, frame_data in enumerate(frames_data):
        frame = frame_data['frame']
        overlay = frame.copy()

        # 1. 繪製 ROI 和 Tripwire 等靜態疊加層
        if Config.ROI_ENABLED and Config.ROI_POLYGON_OBJECT:
            roi_points = np.array(Config.ROI_POLYGON_OBJECT.exterior.coords, dtype=np.int32)
            cv2.fillPoly(overlay, [roi_points], color=(255, 255, 0))
            cv2.polylines(overlay, [roi_points], isClosed=True, color=(255, 255, 0), thickness=2)

        if Config.TRIPWIRES_ENABLED and Config.TRIPWIRE_LINE_OBJECTS:
            line_thickness, tip_length = 4, 0.05
            for tripwire_obj in Config.TRIPWIRE_LINE_OBJECTS:
                line = tripwire_obj["line"]
                direction = tripwire_obj["direction"]
                p1_coords, p2_coords = line.coords[0], line.coords[1]
                p1, p2 = (int(p1_coords[0]), int(p1_coords[1])), (int(p2_coords[0]), int(p2_coords[1]))

                if direction == "cross_to_right":
                    cv2.arrowedLine(overlay, p1, p2, (0, 0, 255), line_thickness, tipLength=tip_length)
                elif direction == "cross_to_left":
                    cv2.arrowedLine(overlay, p2, p1, (0, 0, 255), line_thickness, tipLength=tip_length)
                else:
                    cv2.line(overlay, p1, p2, (0, 0, 255), line_thickness)

        # 2. 繪製追蹤物件
        current_frame_alert_ids = frame_data.get('tripwire_alert_ids', set())
        track_roi_status = frame_data.get('track_roi_status', {})

        # 找出所有活躍的目標ID
        active_track_ids = current_frame_alert_ids.union(
            {tid for tid, in_roi in track_roi_status.items() if in_roi}
        )

        for track_info in frame_data.get('tracks', []):
            box, track_id = track_info['box_xyxy'], track_info['track_id']
            x1, y1, x2, y2 = map(int, box)

            is_active = track_id in active_track_ids

            # 根據目標是否活躍決定顏色和透明度
            if is_active:
                alpha = 1.0
                if track_id in current_frame_alert_ids:
                    box_color = (0, 0, 255)  # 觸發警戒線：紅色
                else:
                    box_color = (0, 255, 255)  # 在 ROI 內：黃色
            else:
                alpha = 0.5
                box_color = (128, 128, 128)  # 非活躍目標：灰色

            # 創建一個用於繪製半透明框的臨時疊加層
            track_overlay = overlay.copy()

            # 繪製 BBox
            cv2.rectangle(track_overlay, (x1, y1), (x2, y2), box_color, 2)

            # 繪製錨點
            anchor_coords = track_info.get('anchors', [])
            for anchor in anchor_coords:
                center_point = (int(anchor[0]), int(anchor[1]))
                cv2.circle(track_overlay, center_point, 5, box_color, -1)

            # 將帶有透明度的 track_overlay 融合回主 overlay
            overlay = cv2.addWeighted(track_overlay, alpha, overlay, 1 - alpha, 0)

            # 繪製文字 (文字不使用透明效果以保證可讀性)
            cv2.putText(overlay, f'ID:{track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        # 3. 將包含所有繪圖的 overlay 與原始幀融合
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # 4. 繪製狀態資訊疊加層
        info_panel = np.zeros((80, width, 3), dtype=np.uint8)
        frame[0:80, 0:width] = cv2.addWeighted(frame[0:80, 0:width], 0.5, info_panel, 0.5, 0)
        event_text = f"EVENT: {event_type.upper()}"
        cv2.putText(frame, event_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elapsed_time = frame_data['time'] - start_time
        duration_text = f"DURATION: {elapsed_time:.1f}s"
        cv2.putText(frame, duration_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if process.stdin:
            process.stdin.write(frame.tobytes())

    if process.stdin:
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