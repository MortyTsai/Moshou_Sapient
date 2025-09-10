# config.py
import os
import logging
from dotenv import load_dotenv
from .utils.video_utils import get_video_resolution

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
    CAPTURES_DIR = os.path.join(PROJECT_ROOT, "captures")
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yolo11s.engine')
    REID_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yolo11s-cls.pt')
    TRACKER_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'custom_botsort.yaml')

    # --- Re-ID 相關參數 ---
    PERSON_MATCH_THRESHOLD = 0.94

    # --- 事件錄影參數 ---
    PRE_EVENT_SECONDS = 2.0
    POST_EVENT_SECONDS = 5.0
    COOLDOWN_PERIOD = 5.0
    TARGET_FPS = 30.0

    # --- 影像尺寸 (預設值) ---
    ENCODE_WIDTH = 2304
    ENCODE_HEIGHT = 1296
    ANALYSIS_WIDTH = 1280
    ANALYSIS_HEIGHT = 736

    # --- 系統參數 ---
    THREAD_JOIN_TIMEOUT = 10
    HEALTH_CHECK_INTERVAL = 15

    @staticmethod
    def ensure_captures_dir_exists():
        os.makedirs(Config.CAPTURES_DIR, exist_ok=True)

    @staticmethod
    def initialize_dynamic_settings():
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