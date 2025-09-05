# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    中央設定類別, 統一管理所有參數與環境變數。
    """
    # Discord Bot
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID")) if os.getenv("DISCORD_CHANNEL_ID") else None

    # Tapo Camera
    TAPO_IP = os.getenv("TAPO_IP")
    TAPO_USER = os.getenv("TAPO_USER")
    TAPO_PASS = os.getenv("TAPO_PASS")
    RTSP_URL_HIGH_RES = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{TAPO_IP}:554/stream1" if all([TAPO_USER, TAPO_PASS, TAPO_IP]) else None

    # 專案核心設定
    CAPTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
    MODEL_PATH = 'yolo11n.engine'

    # 事件錄影參數
    PRE_EVENT_SECONDS = 2.0
    POST_EVENT_SECONDS = 5.0
    COOLDOWN_PERIOD = 5.0
    TARGET_FPS = 30.0

    # 影像尺寸
    ENCODE_WIDTH = 2304
    ENCODE_HEIGHT = 1296
    ANALYSIS_WIDTH = 1280
    ANALYSIS_HEIGHT = 736

    # 系統參數
    THREAD_JOIN_TIMEOUT = 10
    HEALTH_CHECK_INTERVAL = 15

    @staticmethod
    def ensure_captures_dir_exists():
        os.makedirs(Config.CAPTURES_DIR, exist_ok=True)