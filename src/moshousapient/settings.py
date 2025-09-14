# src/moshousapient/settings.py

"""
專案設定模組

此模組集中管理所有可由使用者調整的應用程式參數。
設定會優先從專案根目錄下的 .env 檔案讀取，若 .env 檔案中未定義，則會使用此處指定的預設值。
"""
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# 專案根目錄 (MoshouSapient/)，此為系統自動計算路徑，請勿修改。
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """
    應用程式的核心設定類別。
    """
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding='utf-8',
        case_sensitive=False
    )

    # --- 影像來源設定 ---
    # 影像來源類型。可選值為 "RTSP" 或 "FILE"。
    # "RTSP": 連接至網路攝影機的即時影像串流。
    # "FILE": 播放並分析一個本地影片檔案。
    VIDEO_SOURCE_TYPE: str = "RTSP"

    # 【FILE 模式專用】影片檔案的路徑。
    # 當 VIDEO_SOURCE_TYPE 設定為 "FILE" 時，系統將會讀取此路徑的影片。
    # 範例: "videos/my_test_video.mp4"
    VIDEO_FILE_PATH: Optional[str] = "data/video_samples/your_test_video.mp4"

    # 【RTSP 模式專用】攝影機的 RTSP 串流網址。
    # 請務必填寫完整且正確的網址，包含使用者名稱、密碼與 IP 位址。
    # 範例: "rtsp://admin:password123@192.168.1.100:554/stream1"
    RTSP_URL: Optional[str] = None

    # --- Discord Bot 通知設定 ---
    # 是否啟用 Discord 通知功能。設定為 True 可在偵測到事件時發送訊息。
    DISCORD_ENABLED: bool = False

    # 【Discord 啟用時必須】您的 Discord Bot Token。
    # 請至 Discord 開發者後台取得此金鑰。
    DISCORD_TOKEN: Optional[str] = None

    # 【Discord 啟用時必須】希望接收通知的 Discord 頻道 ID。
    # 請在 Discord 中對目標頻道點擊右鍵，選擇「複製頻道 ID」。
    DISCORD_CHANNEL_ID: Optional[int] = None

    # --- Re-ID (人物重識別) 相關參數 ---
    # 人物特徵比對的相似度閾值。數值範圍為 0 到 1。
    # 數值越高，代表系統對於「判定為同一個人」的要求越嚴格，可減少誤判，但可能將同一個人誤認為不同人。
    # 數值越低，則越寬鬆，容錯率較高，但可能會將不同人誤判為同一人。
    # 建議值: 0.96
    PERSON_MATCH_THRESHOLD: float = 0.96

    # --- 事件錄影參數 ---
    # 事件觸發「前」額外錄製的秒數。
    # 這能確保錄影內容包含事件發生前的完整上下文。
    PRE_EVENT_SECONDS: float = 2.0

    # 當畫面中已無人物時，系統會再持續錄影的秒數。
    # 這能確保錄影內容包含人物離開後的完整畫面。
    POST_EVENT_SECONDS: float = 5.0

    # 一次事件錄影結束後，必須等待的冷卻時間（秒），才能觸發下一次新的事件。
    # 這能有效防止因人物短暫停留或重複進出而產生大量零碎的錄影檔案。
    COOLDOWN_PERIOD: float = 5.0

    # 單一事件錄影的最長持續時間（秒）。
    # 這是一個安全機制，防止因意外情況導致錄影程序無法正常結束，從而產生過大的影片檔案。
    MAX_EVENT_DURATION: float = 20.0

    # --- 事件影片幀率設定 - --
    # 輸出影片的幀率模式。可選值: "TARGET", "SOURCE"
    # "TARGET": 系統會將影片降採樣至下方設定的 TARGET_FPS，有助於節省儲存空間。 (推薦)
    # "SOURCE": 系統會保留影片的原始幀率。
    VIDEO_FPS_MODE: str = "SOURCE"

    # 在 VIDEO_FPS_MODE 設定為 "TARGET" 時，所使用的目標幀率 (FPS)。
    # 較高的 FPS 會使影片更流暢，但檔案大小也會更大。
    TARGET_FPS: float = 30.0

    # --- 事件影片編碼設定 ---
    # 輸出影片的編碼模式。可選值: "QUALITY", "BALANCED"
    # "QUALITY":  恆定品質模式。優先保證每一幀的視覺品質，但檔案大小會因畫面複雜度而大幅波動。(預設)
    # "BALANCED": 均衡模式。將影片的平均位元率控制在一個目標值附近，檔案大小更可預測，適合絕大多數場景。
    VIDEO_ENCODING_MODE: str = "BALANCED"

    # 在編碼模式為 "BALANCED" 時，所使用的目標平均位元率 (單位：Mbps)。
    # 較高的值會帶來更好的畫質和更大的檔案大小。對於 1080p 影片，2-4 Mbps 是一個合理的範圍。
    TARGET_BITRATE_MBPS: float = 2.0

    # --- 影像尺寸設定 ---
    # 最終儲存的影片檔案的解析度（寬度）。
    ENCODE_WIDTH: int = 2304
    # 最終儲存的影片檔案的解析度（高度）。
    ENCODE_HEIGHT: int = 1296

    # 進行 AI 分析時所使用的影像解析度（寬度）。
    # 較低的解析度可以顯著提升 AI 的處理速度，但可能會犧牲一些偵測的精準度。
    # 此設定值應與您訓練或轉換 TensorRT 模型時所用的尺寸一致。
    ANALYSIS_WIDTH: int = 1280
    # 進行 AI 分析時所使用的影像解析度（高度）。
    ANALYSIS_HEIGHT: int = 736

    # --- 系統內部參數 (通常不需修改) ---
    THREAD_JOIN_TIMEOUT: int = 10
    HEALTH_CHECK_INTERVAL: int = 15

    # --- 系統自動生成路徑 (請勿手動修改) ---
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CAPTURES_DIR: Path = DATA_DIR / "captures"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
    DB_FILE: Path = DATA_DIR / "security_events.db"
    MODEL_PATH: Path = MODELS_DIR / "yolo11s.engine"
    REID_MODEL_PATH: Path = MODELS_DIR / "yolo11s-cls.pt"
    TRACKER_CONFIG_PATH: Path = CONFIGS_DIR / "custom_botsort.yaml"
    # 新增：行為分析設定檔的路徑
    BEHAVIOR_CONFIG_PATH: Path = CONFIGS_DIR / "behavior_analysis.yaml"


# 建立一個全域可用的 settings 實例，供應用程式其他部分導入。
settings = Settings()

# 確保應用程式啟動時，存放錄影檔案的目錄已存在。
os.makedirs(settings.CAPTURES_DIR, exist_ok=True)