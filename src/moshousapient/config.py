# src/moshousapient/config.py

import logging
import yaml
from pathlib import Path
from typing import Union

from shapely.geometry import Polygon, LineString
from shapely.errors import ShapelyError

from .settings import settings, PROJECT_ROOT
from .utils.video_utils import get_video_resolution


class Config:
    """
    中央設定類別，統一管理所有參數與應用程式邏輯。
    此類別從 settings 模組讀取原始設定值，並從 YAML 檔案載入複雜的行為分析規則，
    最後執行應用程式啟動時的初始化邏輯。
    """
    # --- 從 settings 模組讀取靜態設定 ---
    VIDEO_SOURCE_TYPE = settings.VIDEO_SOURCE_TYPE.upper()
    VIDEO_FILE_PATH = settings.VIDEO_FILE_PATH
    RTSP_URL = settings.RTSP_URL
    DISCORD_ENABLED = settings.DISCORD_ENABLED
    DISCORD_TOKEN = settings.DISCORD_TOKEN
    DISCORD_CHANNEL_ID = settings.DISCORD_CHANNEL_ID
    PERSON_MATCH_THRESHOLD = settings.PERSON_MATCH_THRESHOLD
    PRE_EVENT_SECONDS = settings.PRE_EVENT_SECONDS
    POST_EVENT_SECONDS = settings.POST_EVENT_SECONDS
    COOLDOWN_PERIOD = settings.COOLDOWN_PERIOD
    TARGET_FPS = settings.TARGET_FPS
    MAX_EVENT_DURATION = settings.MAX_EVENT_DURATION
    THREAD_JOIN_TIMEOUT = settings.THREAD_JOIN_TIMEOUT
    HEALTH_CHECK_INTERVAL = settings.HEALTH_CHECK_INTERVAL

    # --- 路徑設定 ---
    CAPTURES_DIR = str(settings.CAPTURES_DIR)
    MODEL_PATH = str(settings.MODEL_PATH)
    REID_MODEL_PATH = str(settings.REID_MODEL_PATH)
    TRACKER_CONFIG_PATH = str(settings.TRACKER_CONFIG_PATH)
    BEHAVIOR_CONFIG_PATH = str(settings.BEHAVIOR_CONFIG_PATH)

    # --- 動態設定 ---
    ENCODE_WIDTH = settings.ENCODE_WIDTH
    ENCODE_HEIGHT = settings.ENCODE_HEIGHT
    ANALYSIS_WIDTH = settings.ANALYSIS_WIDTH
    ANALYSIS_HEIGHT = settings.ANALYSIS_HEIGHT

    # --- 行為分析參數 (將從 YAML 載入) ---
    ROI_ENABLED: bool = False
    ROI_POLYGON_POINTS: list = []
    ROI_DWELL_TIME_THRESHOLD: float = 3.0
    ROI_POLYGON_OBJECT: Union[Polygon, None] = None

    TRIPWIRES_ENABLED: bool = False
    TRIPWIRE_CONFIGS: list = []
    TRIPWIRE_LINE_OBJECTS: list = []

    # --- 類別方法 (初始化邏輯) ---
    @staticmethod
    def _load_behavior_config():
        """從 behavior_analysis.yaml 載入 ROI 和 Tripwire 設定。"""
        try:
            with open(Config.BEHAVIOR_CONFIG_PATH, 'r', encoding='utf-8') as f:
                behavior_config = yaml.safe_load(f)

            # 載入 ROI 設定
            roi_settings = behavior_config.get('roi', {})
            if roi_settings and roi_settings.get('enabled', False):
                Config.ROI_ENABLED = True
                Config.ROI_POLYGON_POINTS = roi_settings.get('polygon_points', [])
                Config.ROI_DWELL_TIME_THRESHOLD = roi_settings.get('dwell_time_threshold', 3.0)
                logging.info("[系統] 已成功載入 ROI 設定。")

            # 載入 Tripwire 設定
            tripwire_settings = behavior_config.get('tripwires', {})
            if tripwire_settings and tripwire_settings.get('enabled', False):
                Config.TRIPWIRES_ENABLED = True
                Config.TRIPWIRE_CONFIGS = tripwire_settings.get('lines', [])
                logging.info("[系統] 已成功載入 Tripwires 設定。")

        except FileNotFoundError:
            logging.warning(f"[系統] 找不到行為分析設定檔: {Config.BEHAVIOR_CONFIG_PATH}。將停用 ROI 和 Tripwire 功能。")
        except yaml.YAMLError as e:
            logging.error(f"[系統] 解析行為分析設定檔時發生錯誤: {e}。將停用 ROI 和 Tripwire 功能。")

    @staticmethod
    def _initialize_roi():
        """根據載入的設定，初始化 Shapely Polygon 物件。"""
        if not Config.ROI_ENABLED:
            logging.info("[系統] ROI 功能未啟用，已跳過初始化。")
            Config.ROI_POLYGON_OBJECT = None
            return

        if Config.ROI_POLYGON_POINTS and len(Config.ROI_POLYGON_POINTS) >= 3:
            try:
                Config.ROI_POLYGON_OBJECT = Polygon(Config.ROI_POLYGON_POINTS)
                logging.info(f"[系統] 成功建立 ROI 區域，面積: {Config.ROI_POLYGON_OBJECT.area} 平方像素。")
            except (ShapelyError, TypeError) as e:
                logging.warning(f"[系統] 無法建立 ROI 區域，設定的座標點可能無效: {e}。ROI 功能將被停用。")
                Config.ROI_POLYGON_OBJECT = None
        else:
            logging.info("[系統] 未設定有效的 ROI 區域或座標點少於 3 個，ROI 功能已停用。")
            Config.ROI_POLYGON_OBJECT = None

    @staticmethod
    def _initialize_tripwires():
        """根據載入的設定，初始化所有警戒線物件。"""
        Config.TRIPWIRE_LINE_OBJECTS.clear()
        if not Config.TRIPWIRES_ENABLED:
            logging.info("[系統] Tripwire 功能未啟用，已跳過初始化。")
            return

        if Config.TRIPWIRE_CONFIGS:
            for config in Config.TRIPWIRE_CONFIGS:
                try:
                    points = config.get("points")
                    direction = config.get("alert_direction", "both")
                    if not points or len(points) != 2:
                        logging.warning(f"[系統] 警戒線定義無效 (需要 2 個點)，已跳過: {config}")
                        continue
                    line = LineString(points)
                    Config.TRIPWIRE_LINE_OBJECTS.append({"line": line, "direction": direction})
                except (ShapelyError, TypeError, KeyError) as e:
                    logging.warning(f"[系統] 無法建立警戒線，設定可能無效: {e}。已跳過該設定: {config}")

        if Config.TRIPWIRE_LINE_OBJECTS:
            logging.info(f"[系統] 成功建立 {len(Config.TRIPWIRE_LINE_OBJECTS)} 條方向性感測警戒線。")
        else:
            logging.info("[系統] 未設定任何有效的虛擬警戒線。")

    @staticmethod
    def initialize_dynamic_settings():
        """初始化那些可能需要根據環境或檔案內容動態決定的設定。"""
        if Config.VIDEO_SOURCE_TYPE == "FILE":
            logging.info("[系統] 偵測到檔案模式，正在動態獲取影片解析度...")
            video_path_str = Config.VIDEO_FILE_PATH
            if not video_path_str:
                logging.warning("[系統] 檔案模式已啟用，但未提供 VIDEO_FILE_PATH。")
                return

            video_path = Path(video_path_str)
            if not video_path.is_absolute():
                video_path = PROJECT_ROOT / video_path

            if video_path.exists():
                resolution = get_video_resolution(str(video_path))
                if resolution:
                    Config.ENCODE_WIDTH, Config.ENCODE_HEIGHT = resolution
                    logging.info(f"[系統] 已動態更新影像尺寸為: {resolution[0]}x{resolution[1]}")
                else:
                    logging.error("[系統] 無法獲取影片解析度，將使用預設值。")
            else:
                logging.warning(f"[系統] 未找到有效的影片檔案: {video_path}，將使用預設影像尺寸。")

    @staticmethod
    def initialize_static_settings():
        """執行所有在模組載入時就應完成的靜態設定初始化。"""
        Config._load_behavior_config()
        Config._initialize_roi()
        Config._initialize_tripwires()