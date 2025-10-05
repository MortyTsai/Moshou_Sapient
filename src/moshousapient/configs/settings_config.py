# src/moshousapient/configs/settings_config.py
"""
專案設定模組，使用 Pydantic-Settings 實現類型安全的設定管理。

此模組集中管理所有可由使用者透過 .env 檔案調整的應用程式參數。
設定會優先從專案根目錄下的 .env 檔案讀取，若 .env 檔案中未定義，
則會使用此處指定的預設值。
"""

# 1. 標準庫導入
import os
from pathlib import Path
from typing import Optional

# 2. 第三方庫導入
from pydantic_settings import BaseSettings, SettingsConfigDict

# 專案根目錄 (MoshouSapient/)，此為系統自動計算路徑，請勿修改。
# Path(__file__) -> .../src/moshousapient/configs/settings_config.py
# .parent -> .../src/moshousapient/configs/
# .parent -> .../src/moshousapient/
# .parent -> .../src/
# .parent -> .../ (Project Root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """
    應用程式的核心設定類別，定義了所有環境變數及其類型和預設值。
    """
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding='utf-8',
        case_sensitive=False
    )

    # --- 影像來源設定 ---
    VIDEO_SOURCE_TYPE: str = "RTSP"
    VIDEO_FILE_PATH: Optional[str] = "data/video_samples/input.mp4"
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


# 建立一個全域可用的 settings 實例，供應用程式其他部分導入。
settings = Settings()

# 確保應用程式啟動時，存放錄影檔案的目錄已存在。
os.makedirs(settings.CAPTURES_DIR, exist_ok=True)