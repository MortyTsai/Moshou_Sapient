# src/moshousapient/configs/settings_config.py
"""
專案設定模組，使用 Pydantic-Settings 實現類型安全的設定管理。
"""

# 1. 標準庫導入
import os
from pathlib import Path
from typing import Optional

# 2. 第三方庫導入
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", case_sensitive=False)

    # --- 系統運行模式設定 ---
    VIDEO_SOURCE_TYPE: str = "RTSP"
    RTSP_URL: Optional[str] = None
    RTSP_TRANSPORT_PROTOCOL: str = "UDP"

    # --- Discord Bot 通知設定 ---
    DISCORD_ENABLED: bool = False
    DISCORD_TOKEN: Optional[str] = None
    DISCORD_CHANNEL_ID: Optional[int] = None

    # --- Re-ID (人物重識別) 相關參數 ---
    PERSON_MATCH_THRESHOLD: float = 0.96

    # --- 事件錄影參數 ---
    PRE_EVENT_SECONDS: float = 1.0
    POST_EVENT_SECONDS: float = 3.0
    COOLDOWN_PERIOD: float = 5.0
    MAX_EVENT_DURATION: float = 10.0

    # --- 事件影片幀率設定 ---
    VIDEO_FPS_MODE: str = "SOURCE"
    TARGET_FPS: float = 30.0

    # --- 事件影片編碼設定 ---
    VIDEO_ENCODING_MODE: str = "BALANCED"
    TARGET_BITRATE_MBPS: float = 2.0

    # --- 影像尺寸設定 ---
    ENCODE_WIDTH: int = 1920
    ENCODE_HEIGHT: int = 1080
    ANALYSIS_WIDTH: int = 1280
    ANALYSIS_HEIGHT: int = 736

    # --- 視覺化設定 ---
    TRIPWIRE_LINE_THICKNESS: int = 6
    TRIPWIRE_TIP_LENGTH: float = 0.03

    # --- 系統內部參數 (通常不需修改) ---
    THREAD_JOIN_TIMEOUT: int = 10
    HEALTH_CHECK_INTERVAL: int = 15
    VIDEO_PROCESSING_WORKERS: int = 2

    # --- 日誌系統設定 ---
    LOG_LEVEL: str = "INFO"

    # --- 檔案攝取服務設定 ---
    INGESTION_ENABLED: bool = False
    INGESTION_WATCH_DIR: Path = PROJECT_ROOT / "data" / "uploads"

    # --- 智慧排程器設定 ---
    SCHEDULER_ENABLED: bool = False
    SCHEDULER_CHECK_INTERVAL: int = 15
    SCHEDULER_TASK_RESCUE_TIMEOUT: int = 1800  # 任務救援超時 (秒)，預設 30 分鐘

    # --- 系統自動生成路徑 (請勿手動修改) ---
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CAPTURES_DIR: Path = DATA_DIR / "captures"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
    DB_FILE: Path = DATA_DIR / "security_events.db"
    MODEL_PATH: Path = MODELS_DIR / "yolo11s.engine"
    REID_MODEL_PATH: Path = MODELS_DIR / "yolo11s-cls.pt"
    TRACKER_CONFIG_PATH: Path = CONFIGS_DIR / "custom_botsort.yaml"
    BEHAVIOR_CONFIG_PATH: Path = CONFIGS_DIR / "behavior_analysis.yaml"


settings = Settings()
os.makedirs(settings.CAPTURES_DIR, exist_ok=True)
os.makedirs(settings.INGESTION_WATCH_DIR, exist_ok=True)
