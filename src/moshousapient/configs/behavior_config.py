# src/moshousapient/configs/behavior_config.py
"""
中央設定模組，用於統一管理所有參數與應用程式的啟動邏輯。

此模組從 settings_config 模組讀取原始設定值，並從 YAML 檔案載入複雜的行為分析規則，
最後執行應用程式啟動時的初始化邏輯。
"""

# 1. 標準庫導入
import logging
import yaml
from typing import Union, List, Dict, Any

# 2. 第三方庫導入
from shapely.geometry import Polygon, LineString
from shapely.errors import ShapelyError

# 3. 本專案相對導入
from .settings_config import settings


class Config:
    """
    一個靜態類別，作為所有應用程式配置的中央存取點。
    """
    # --- 從 settings 模組讀取靜態設定 ---
    VIDEO_SOURCE_TYPE: str = settings.VIDEO_SOURCE_TYPE.upper()
    VIDEO_FILE_PATH: str = settings.VIDEO_FILE_PATH
    RTSP_URL: str = settings.RTSP_URL
    RTSP_TRANSPORT_PROTOCOL: str = settings.RTSP_TRANSPORT_PROTOCOL.upper()
    DISCORD_ENABLED: bool = settings.DISCORD_ENABLED
    DISCORD_TOKEN: str = settings.DISCORD_TOKEN
    DISCORD_CHANNEL_ID: int = settings.DISCORD_CHANNEL_ID
    PERSON_MATCH_THRESHOLD: float = settings.PERSON_MATCH_THRESHOLD
    PRE_EVENT_SECONDS: float = settings.PRE_EVENT_SECONDS
    POST_EVENT_SECONDS: float = settings.POST_EVENT_SECONDS
    COOLDOWN_PERIOD: float = settings.COOLDOWN_PERIOD
    MAX_EVENT_DURATION: float = settings.MAX_EVENT_DURATION
    VIDEO_FPS_MODE: str = settings.VIDEO_FPS_MODE.upper()
    TARGET_FPS: float = settings.TARGET_FPS
    VIDEO_ENCODING_MODE: str = settings.VIDEO_ENCODING_MODE.upper()
    TARGET_BITRATE_MBPS: float = settings.TARGET_BITRATE_MBPS
    THREAD_JOIN_TIMEOUT: int = settings.THREAD_JOIN_TIMEOUT
    HEALTH_CHECK_INTERVAL: int = settings.HEALTH_CHECK_INTERVAL
    TRIPWIRE_LINE_THICKNESS: int = settings.TRIPWIRE_LINE_THICKNESS
    TRIPWIRE_TIP_LENGTH: float = settings.TRIPWIRE_TIP_LENGTH

    # --- 路徑設定 ---
    CAPTURES_DIR: str = str(settings.CAPTURES_DIR)
    MODEL_PATH: str = str(settings.MODEL_PATH)
    REID_MODEL_PATH: str = str(settings.REID_MODEL_PATH)
    TRACKER_CONFIG_PATH: str = str(settings.TRACKER_CONFIG_PATH)
    BEHAVIOR_CONFIG_PATH: str = str(settings.BEHAVIOR_CONFIG_PATH)

    # --- 動態設定 (將由 app_orchestrator.py 初始化) ---
    ENCODE_WIDTH: int = settings.ENCODE_WIDTH
    ENCODE_HEIGHT: int = settings.ENCODE_HEIGHT
    ANALYSIS_WIDTH: int = settings.ANALYSIS_WIDTH
    ANALYSIS_HEIGHT: int = settings.ANALYSIS_HEIGHT

    # --- 行為分析參數 (將從 YAML 載入) ---
    # 錨點系統設定
    ANCHOR_POINTS: Union[str, List[str]] = 'bottom_center'
    # ROI 相關設定
    ROI_ENABLED: bool = False
    ROI_SETTINGS: Dict[str, Any] = {}
    ROI_POLYGON_OBJECT: Union[Polygon, None] = None
    # Tripwire 相關設定
    TRIPWIRES_ENABLED: bool = False
    TRIPWIRE_SETTINGS: Dict[str, Any] = {}
    TRIPWIRE_LINE_OBJECTS: List[Dict[str, Any]] = []
    # 遮蔽警報設定
    OCCLUSION_ALERT_ENABLED: bool = False
    OCCLUSION_ALERT_SETTINGS: Dict[str, Any] = {}
    # 畫面異常警報設定
    SCENE_ANOMALY_ALERT_ENABLED: bool = False
    SCENE_ANOMALY_ALERT_SETTINGS: Dict[str, Any] = {}

    @staticmethod
    def _load_behavior_config():
        """從 behavior_analysis.yaml 載入所有行為分析規則。"""
        try:
            with open(Config.BEHAVIOR_CONFIG_PATH, 'r', encoding='utf-8') as f:
                behavior_config = yaml.safe_load(f) or {}

            # 1. 載入全域錨點設定 (提供預設值以確保相容性)
            Config.ANCHOR_POINTS = behavior_config.get('anchor_points', 'bottom_center')

            # 2. 載入 ROI 設定
            Config.ROI_SETTINGS = behavior_config.get('roi', {})
            if Config.ROI_SETTINGS.get('enabled', False):
                Config.ROI_ENABLED = True
                logging.debug("[系統] 已成功載入 ROI 設定。")

            # 3. 載入 Tripwire 設定
            Config.TRIPWIRE_SETTINGS = behavior_config.get('tripwires', {})
            if Config.TRIPWIRE_SETTINGS.get('enabled', False):
                Config.TRIPWIRES_ENABLED = True
                logging.debug("[系統] 已成功載入 Tripwires 設定。")

            # 4. 載入遮蔽警報設定
            Config.OCCLUSION_ALERT_SETTINGS = behavior_config.get('occlusion_alerts', {})
            if Config.OCCLUSION_ALERT_SETTINGS.get('enabled', False):
                Config.OCCLUSION_ALERT_ENABLED = True
                logging.debug("[系統] 已成功載入 Occlusion Alert 設定。")

            # 5. 載入畫面異常警報設定
            Config.SCENE_ANOMALY_ALERT_SETTINGS = behavior_config.get('scene_anomaly_alerts', {})
            if Config.SCENE_ANOMALY_ALERT_SETTINGS.get('enabled', False):
                Config.SCENE_ANOMALY_ENABLED = True
                logging.debug("[系統] 已成功載入 Scene Anomaly Alert 設定。")

        except FileNotFoundError:
            logging.warning(f"[系統] 找不到行為分析設定檔: {Config.BEHAVIOR_CONFIG_PATH}。將停用所有高階行為分析功能。")
        except yaml.YAMLError as e:
            logging.error(f"[系統] 解析行為分析設定檔時發生錯誤: {e}。將停用所有高階行為分析功能。")

    @staticmethod
    def _initialize_roi():
        """根據載入的設定，初始化 Shapely Polygon 物件。"""
        if not Config.ROI_ENABLED:
            logging.debug("[系統] ROI 功能未啟用，已跳過初始化。")
            Config.ROI_POLYGON_OBJECT = None
            return

        polygon_points = Config.ROI_SETTINGS.get('polygon_points', [])
        if polygon_points and len(polygon_points) >= 3:
            try:
                Config.ROI_POLYGON_OBJECT = Polygon(polygon_points)
                logging.debug(f"[系統] 成功建立 ROI 區域，面積: {Config.ROI_POLYGON_OBJECT.area:.2f} 平方像素。")
            except (ShapelyError, TypeError) as e:
                logging.warning(f"[系統] 無法建立 ROI 區域，設定的座標點可能無效: {e}。ROI 功能將被停用。")
                Config.ROI_POLYGON_OBJECT = None
        else:
            logging.debug("[系統] 未設定有效的 ROI 區域或座標點少於 3 個，ROI 功能已停用。")
            Config.ROI_POLYGON_OBJECT = None

    @staticmethod
    def _initialize_tripwires():
        """根據載入的設定，初始化所有警戒線 Shapely LineString 物件。"""
        Config.TRIPWIRE_LINE_OBJECTS.clear()
        if not Config.TRIPWIRES_ENABLED:
            logging.debug("[系統] Tripwire 功能未啟用，已跳過初始化。")
            return

        lines = Config.TRIPWIRE_SETTINGS.get('lines', [])
        if not lines:
            logging.debug("[系統] 未設定任何有效的虛擬警戒線。")
            return

        for line_config in lines:
            try:
                points = line_config.get("points")
                if not points or len(points) != 2:
                    logging.warning(f"[系統] 警戒線定義無效 (需要 2 個點)，已跳過: {line_config}")
                    continue

                line = LineString(points)
                direction = line_config.get("alert_direction", "both")
                # 儲存解析後的物件和原始設定
                Config.TRIPWIRE_LINE_OBJECTS.append({
                    "line": line,
                    "direction": direction,
                    "config": line_config  # 保留原始設定以供後續使用 (例如讀取錨點覆寫)
                })
            except (ShapelyError, TypeError, KeyError) as e:
                logging.warning(f"[系統] 無法建立警戒線，設定可能無效: {e}。已跳過該設定: {line_config}")

        if Config.TRIPWIRE_LINE_OBJECTS:
            logging.debug(f"[系統] 成功建立 {len(Config.TRIPWIRE_LINE_OBJECTS)} 條方向性感測警戒線。")
        else:
            logging.debug("[系統] 未設定任何有效的虛擬警戒線。")

    @staticmethod
    def initialize_static_settings():
        """
        執行所有在模組載入時就應完成的靜態設定初始化。
        """
        Config._load_behavior_config()
        Config._initialize_roi()
        Config._initialize_tripwires()