# src/moshousapient/config.py

import os
import logging
from dotenv import load_dotenv
from .utils.video_utils import get_video_resolution
from shapely.geometry import Polygon, LineString
from shapely.errors import ShapelyError

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Config:
    """
    中央設定類別，統一管理所有參數與環境變數。
    所有檔案路徑均應建構成絕對路徑以保證穩定性。
    """

    # --- 影像來源設定 ---
    VIDEO_SOURCE_TYPE = os.getenv("VIDEO_SOURCE_TYPE", "RTSP").upper()
    VIDEO_FILE_PATH = os.getenv("VIDEO_FILE_PATH")
    RTSP_URL = os.getenv("RTSP_URL")

    # --- Discord Bot 設定 ---
    DISCORD_ENABLED = os.getenv("DISCORD_ENABLED", "False").lower() == "true"
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID")) if os.getenv("DISCORD_CHANNEL_ID") else None

    # --- 專案核心路徑設定 ---
    CAPTURES_DIR = os.path.join(PROJECT_ROOT, "data", "captures")
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yolo11s.engine')
    REID_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yolo11s-cls.pt')
    TRACKER_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'custom_botsort.yaml')

    # --- Re-ID 相關參數 ---
    PERSON_MATCH_THRESHOLD = 0.96

    # --- 事件錄影參數 ---
    PRE_EVENT_SECONDS = 2.0
    POST_EVENT_SECONDS = 5.0
    COOLDOWN_PERIOD = 5.0
    TARGET_FPS = 30.0
    MAX_EVENT_DURATION = 20.0

    # --- 影像尺寸 (預設值) ---
    ENCODE_WIDTH = 2304
    ENCODE_HEIGHT = 1296
    ANALYSIS_WIDTH = 1280
    ANALYSIS_HEIGHT = 736

    # --- 系統參數 ---
    THREAD_JOIN_TIMEOUT = 10
    HEALTH_CHECK_INTERVAL = 15

    # --- 行為分析參數 (ROI) ---
    ROI_POLYGON_POINTS = [
        [640, 200], [1280, 200], [1280, 720], [640, 720]
    ]
    ROI_POLYGON_OBJECT = None
    ROI_DWELL_TIME_THRESHOLD = 3.0

    # --- 行為分析參數 (Tripwire) --- #
    # 升級結構以支援精確的方向性檢測
    # 結構: [ { "points": [[x1, y1], [x2, y2]], "alert_direction": "cross_to_right" }, ... ]
    # "points": 定義線段的起點(p1)和終點(p2)。線段的方向為 p1 -> p2。
    # "alert_direction": "cross_to_right", "cross_to_left", 或 "both"。
    #   - "cross_to_right": 當移動軌跡從線段向量的「左側」穿越到「右側」時觸發。
    #   - "cross_to_left": 當移動軌跡從線段向量的「右側」穿越到「左側」時觸發。
    #   - "both": 雙向均觸發。
    TRIPWIRE_CONFIGS = [
        {
            "points": [[300, 600], [1000, 600]],  # 一條從左(p1)到右(p2)的水平線
            "alert_direction": "cross_to_right"  # 觸發條件: 從線的上方(左側)穿越到下方(右側)
        }
    ]
    TRIPWIRE_LINE_OBJECTS = []

    # --- 類別方法 ---
    @staticmethod
    def _initialize_roi():
        """
        私有輔助方法，根據座標點初始化 Shapely Polygon 物件。
        """
        if Config.ROI_POLYGON_POINTS and len(Config.ROI_POLYGON_POINTS) >= 3:
            try:
                Config.ROI_POLYGON_OBJECT = Polygon(Config.ROI_POLYGON_POINTS)
                logging.info(f"[系統] 成功建立 ROI 區域，面積: {Config.ROI_POLYGON_OBJECT.area} 平方像素。")
            except (ShapelyError, TypeError) as e:
                logging.warning(f"[系統] 無法建立 ROI 區域，設定的座標點可能無效: {e}。ROI 功能將被停用。")
                Config.ROI_POLYGON_OBJECT = None
        else:
            logging.info("[系統] 未設定有效的 ROI 區域或座標點少於3個，ROI 功能已停用。")
            Config.ROI_POLYGON_OBJECT = None

    @staticmethod
    def _initialize_tripwires():
        """
        私有輔助方法，根據設定檔解析並初始化所有警戒線物件。
        """
        # 清空列表以防重複初始化
        Config.TRIPWIRE_LINE_OBJECTS.clear()

        if Config.TRIPWIRE_CONFIGS:
            for config in Config.TRIPWIRE_CONFIGS:
                try:
                    points = config.get("points")
                    direction = config.get("alert_direction", "both")

                    if not points or len(points) != 2:
                        logging.warning(f"[系統] 警戒線定義無效 (需要2個點)，已跳過: {config}")
                        continue

                    line = LineString(points)

                    # 將 Shapely 物件和方向設定一起存儲
                    Config.TRIPWIRE_LINE_OBJECTS.append({
                        "line": line,
                        "direction": direction
                    })

                except (ShapelyError, TypeError, KeyError) as e:
                    logging.warning(f"[系統] 無法建立警戒線，設定可能無效: {e}。已跳過該設定: {config}")

            if Config.TRIPWIRE_LINE_OBJECTS:
                logging.info(f"[系統] 成功建立 {len(Config.TRIPWIRE_LINE_OBJECTS)} 條方向性感測警戒線。")
        else:
            logging.info("[系統] 未設定任何虛擬警戒線，Tripwire 功能已停用。")

    @staticmethod
    def ensure_captures_dir_exists():
        os.makedirs(Config.CAPTURES_DIR, exist_ok=True)

    @staticmethod
    def initialize_dynamic_settings():
        """
        初始化那些可能需要根據環境或檔案內容動態決定的設定。
        """
        if Config.VIDEO_SOURCE_TYPE == "FILE":
            logging.info("[系統] 偵測到檔案模式，正在動態獲取影片解析度...")
            video_path = Config.VIDEO_FILE_PATH
            if video_path and not os.path.isabs(video_path):
                video_path = os.path.join(PROJECT_ROOT, video_path)

            if video_path and os.path.exists(video_path):
                resolution = get_video_resolution(video_path)
                if resolution:
                    Config.ENCODE_WIDTH, Config.ENCODE_HEIGHT = resolution
                else:
                    logging.error("[系統] 無法獲取影片解析度，將使用預設值。")
            else:
                logging.warning(f"[系統] 未找到有效的影片檔案: {video_path}，將使用預設影像尺寸。")

    @staticmethod
    def initialize_static_settings():
        """
        執行所有在模組載入時就應完成的靜態設定初始化。
        """
        Config._initialize_roi()
        Config._initialize_tripwires()
