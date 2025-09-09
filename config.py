# config.py
import os
import logging
from dotenv import load_dotenv
from utils.video_utils import get_video_resolution

load_dotenv()


class Config:
    """
    中央設定類別, 統一管理所有參數與環境變數。
    """

    # Source Switching
    VIDEO_SOURCE_TYPE = os.getenv("VIDEO_SOURCE_TYPE", "RTSP").upper()
    VIDEO_FILE_PATH = os.getenv("VIDEO_FILE_PATH")

    # Discord Bot
    DISCORD_ENABLED = os.getenv("DISCORD_ENABLED", "False").lower() == "true"
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID")) if os.getenv("DISCORD_CHANNEL_ID") else None

    # Camera
    RTSP_URL = os.getenv("RTSP_URL")

    # 專案核心設定
    CAPTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
    MODEL_PATH = 'yolo11s.engine'

    # 事件錄影參數
    PRE_EVENT_SECONDS = 2.0
    POST_EVENT_SECONDS = 5.0
    COOLDOWN_PERIOD = 5.0
    TARGET_FPS = 30.0

    # --- 影像尺寸 (動態設定) ---
    ENCODE_WIDTH = 2304  # 預設為 2K (RTSP 模式)
    ENCODE_HEIGHT = 1296  # 預設為 2K (RTSP 模式)

    if VIDEO_SOURCE_TYPE == "FILE":
        logging.info(f"[系統] 偵測到檔案模式，正在動態獲取影片解析度...")
        if VIDEO_FILE_PATH and os.path.exists(VIDEO_FILE_PATH):
            resolution = get_video_resolution(VIDEO_FILE_PATH)
            if resolution:
                ENCODE_WIDTH, ENCODE_HEIGHT = resolution
            else:
                logging.error("[系統] 無法獲取影片解析度，將使用預設值。請檢查影片檔案與 ffprobe 安裝。")
        else:
            logging.warning("[系統] 未設定有效的 VIDEO_FILE_PATH，將使用預設影像尺寸。")

    # 分析尺寸通常是固定的，或者可以按比例縮放
    ANALYSIS_WIDTH = 1280
    ANALYSIS_HEIGHT = 736

    # 系統參數
    THREAD_JOIN_TIMEOUT = 10
    HEALTH_CHECK_INTERVAL = 15

    @staticmethod
    def ensure_captures_dir_exists():
        os.makedirs(Config.CAPTURES_DIR, exist_ok=True)