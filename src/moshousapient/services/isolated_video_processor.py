# src/moshousapient/services/isolated_video_processor.py (Definitive Final Cleaned Version 2)
import argparse, logging, sys, subprocess, pickle, os
from pathlib import Path
import cv2
import numpy as np
from moshousapient.config import Config
from moshousapient.utils.visualization_utils import draw_static_overlays, draw_dynamic_overlays, draw_info_panel

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
    處理單個影片事件：讀取幀資料、使用統一的視覺化模組繪製元素、並使用 FFmpeg 硬體編碼儲存影片。
    此函式在一個獨立的子程序中執行。
    """
    logging.info(f"開始處理事件 '{event_type}'，從資料檔案: {input_data_path}")
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
        logging.warning("輸入資料為空，不進行處理。")
        return

    start_time = frames_data[0]['time']
    height, width, _ = frames_data[0]['frame'].shape
    duration = frames_data[-1]['time'] - start_time
    actual_fps = len(frames_data) / duration if duration > 0 else Config.TARGET_FPS

    ffmpeg_encode_cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24', '-r', str(actual_fps), '-i', '-',
        '-c:v', 'hevc_nvenc', '-preset', 'p6', '-an', output_path
    ]

    process = subprocess.Popen(ffmpeg_encode_cmd, stdin=subprocess.PIPE)
    logging.info(f"開始對 {len(frames_data)} 幀影像進行繪圖和硬體編碼...")

    # 預先繪製靜態疊加層 (scale 設為 1.0，因為座標系相同)
    static_overlay_template = np.zeros((height, width, 3), dtype=np.uint8)
    static_overlay_template = draw_static_overlays(
        frame=static_overlay_template,
        scale_x=1.0,
        scale_y=1.0,
        roi_enabled=Config.ROI_ENABLED,
        roi_polygon=Config.ROI_POLYGON_OBJECT,
        tripwires_enabled=Config.TRIPWIRES_ENABLED,
        tripwire_line_objects=Config.TRIPWIRE_LINE_OBJECTS,
        tripwire_line_thickness=Config.TRIPWIRE_LINE_THICKNESS,
        tripwire_tip_length=Config.TRIPWIRE_TIP_LENGTH
    )

    for frame_data in frames_data:
        frame = frame_data['frame']

        # 1. 應用靜態疊加層
        overlay = cv2.add(frame, static_overlay_template)

        # 2. 準備並繪製動態疊加層
        active_alert_ids = frame_data.get('tripwire_alert_ids', set())
        track_roi_status = frame_data.get('track_roi_status', {})
        active_roi_ids = {tid for tid, in_roi in track_roi_status.items() if in_roi}
        all_tracks = frame_data.get('tracks', [])

        # 在 RTSP 模式下，座標系已經是最終尺寸，因此 scale 設為 1.0
        overlay = draw_dynamic_overlays(overlay, all_tracks, active_alert_ids, active_roi_ids, 1.0, 1.0)

        # 3. 將所有繪圖與原始幀融合
        final_frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # 4. 繪製資訊面板
        elapsed_time = frame_data['time'] - start_time
        final_frame = draw_info_panel(final_frame, event_type, elapsed_time)

        if process.stdin:
            process.stdin.write(final_frame.tobytes())

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