# src/moshousapient/core/app_orchestrator.py
"""
MoshouSapient 應用程式的主入口點與協調器。

負責初始化所有組件、根據設定選擇執行策略，並管理應用程式的生命週期。
"""

# 1. 標準庫導入
import logging
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any

# 2. 第三方庫導入
import torch

# 3. 本專案相對導入
# 從新的 configs 套件導入
from ..configs.behavior_config import Config
from ..configs.logging_config import setup_logging
from ..configs.settings_config import settings, PROJECT_ROOT

# 從新的 services 套件導入
from ..services.database_service import init_db
from ..services.notification_service import NotificationService

# 其他導入路徑更新
from ..web.app import create_flask_app
from ..utils.video_io_utils import get_video_resolution
from ..processors.rtsp_processing_pipeline import RTSPPipeline
from .producer_runners import RTSPProducerRunner, FileProducerRunner, BaseRunner
from .worker_manager import WorkerManager


def pre_flight_checks() -> bool:
    """
    執行應用程式啟動前的基本環境和設定檢查。

    :return: 檢查通過返回 True，否則返回 False。
    """
    logging.info("[系統] 執行啟動前環境檢查...")
    if Config.VIDEO_SOURCE_TYPE == "RTSP":
        if not torch.cuda.is_available():
            logging.critical("-" * 60)
            logging.critical("[嚴重錯誤] PyTorch 無法偵測到任何可用的 CUDA 設備。")
            logging.critical("請確認:")
            logging.critical("  1. NVIDIA 驅動程式已正確安裝。")
            logging.critical("  2. 您已安裝支援 GPU 的 PyTorch 版本 (版本號不應包含 '+cpu')。")
            logging.critical("  請執行 'pip uninstall torch' 後，參考 PyTorch 官網安裝 GPU 版本。")
            logging.critical("-" * 60)
            return False
        logging.info(f"[系統] CUDA 設備檢查通過。偵測到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("[系統] 在 FILE 模式下，跳過 CUDA 設備檢查。")
    return True


def get_camera_config() -> Optional[Dict[str, Any]]:
    """
    根據全域設定產生單個攝影機的設定字典。

    :return: 包含攝影機設定的字典，如果設定無效則返回 None。
    """
    if Config.VIDEO_SOURCE_TYPE == "RTSP":
        if not Config.RTSP_URL:
            logging.critical("[嚴重錯誤] 未設定完整的 RTSP_URL，請檢查 .env 檔案。")
            return None
        logging.info(f"[系統] 影像來源模式: RTSP 即時串流")
        source_uri = Config.RTSP_URL
        source_name = "RTSP-Cam"
        protocol_setting = Config.RTSP_TRANSPORT_PROTOCOL.lower()
        if protocol_setting not in ["udp", "tcp"]:
            logging.warning(f"[設定警告] 無效的 RTSP_TRANSPORT_PROTOCOL: "
                            f"'{protocol_setting}'。將使用預設值 'udp'。")
            transport_protocol = "udp"
        else:
            transport_protocol = protocol_setting
        return {
            "name": f"Pipeline-{source_name}",
            "rtsp_url": source_uri,
            "transport_protocol": transport_protocol
        }
    return None


def main():
    """
    應用程式主入口點。
    """
    # 1. 基礎初始化 (日誌和設定)
    setup_logging()
    Config.initialize_static_settings()

    if not pre_flight_checks():
        sys.exit(1)

    init_db()

    # 2. 初始化通知器 (所有模式共用)
    notifier: Optional[NotificationService] = None
    if Config.DISCORD_ENABLED:
        if Config.DISCORD_TOKEN and Config.DISCORD_CHANNEL_ID:
            notifier = NotificationService(token=Config.DISCORD_TOKEN,
                                           channel_id=Config.DISCORD_CHANNEL_ID)
            notifier.start()
        else:
            logging.warning("[系統] Discord 功能已啟用，但未提供完整的憑證。通知功能將被禁用。")
    else:
        logging.info("[系統] Discord 通知功能已被禁用。")

    # 3. 啟動 Web 儀表板 (所有模式共用)
    logging.info("[系統] 正在背景啟動 Web 儀表板...")
    flask_app = create_flask_app()
    web_thread = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
        daemon=True,
        name="WebDashboardThread"
    )
    web_thread.start()
    logging.info("[系統] Web 儀表板已在 http://127.0.0.1:5000 上運作。")

    # 4. 根據設定選擇並建立執行策略 (Runner)
    runner: Optional[BaseRunner] = None
    if Config.VIDEO_SOURCE_TYPE == "RTSP":
        try:
            from ultralytics import YOLO
            import numpy as np
            logging.info(f"[YOLO] 正在從 {Config.MODEL_PATH} 載入 TensorRT 模型...")
            model = YOLO(Config.MODEL_PATH, task='detect')
            warmup_frame = np.zeros((Config.ANALYSIS_HEIGHT, Config.ANALYSIS_WIDTH, 3), dtype=np.uint8)
            model.predict(warmup_frame, device=0, verbose=False)
            logging.info("[YOLO] TensorRT 模型已成功載入並預熱。")

            logging.info(f"[Re-ID] 正在載入 {Config.REID_MODEL_PATH} 作為特徵提取器...")
            reid_model = YOLO(Config.REID_MODEL_PATH)
            reid_model.embed(warmup_frame, device=0, verbose=False)
            logging.info("[Re-ID] Re-ID 模型已成功載入並預熱。")

            camera_config = get_camera_config()
            if not camera_config:
                if notifier: notifier.stop()
                sys.exit(1)

            pipelines = [RTSPPipeline(camera_config, model, reid_model, notifier)]
            runner = RTSPProducerRunner(pipelines, notifier)

        except Exception as e:
            logging.critical(f"[模型載入] 嚴重錯誤: 無法載入 AI 模型。{e}", exc_info=True)
            if notifier: notifier.stop()
            sys.exit(1)

    elif Config.VIDEO_SOURCE_TYPE == "FILE":
        video_path_str = Config.VIDEO_FILE_PATH
        if not video_path_str:
            logging.warning("[系統] 檔案模式已啟用，但未提供 VIDEO_FILE_PATH。")
        else:
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

        runner = FileProducerRunner(workers=[], notifier=notifier)  # workers 參數已廢棄

    else:
        logging.critical(
            f"[嚴重錯誤] 無效的 VIDEO_SOURCE_TYPE: '{Config.VIDEO_SOURCE_TYPE}'。"
            f"請在 .env 中設定為 'RTSP' 或 'FILE'。"
        )
        if notifier: notifier.stop()
        sys.exit(1)

    # 5. 啟動 Worker 池並執行主邏輯
    worker_manager = WorkerManager(num_workers=settings.VIDEO_PROCESSING_WORKERS)
    try:
        if runner:
            worker_manager.start_workers()
            runner.run()
        else:
            logging.error("[系統] 未能建立有效的執行器，系統即將關閉。")
    except (KeyboardInterrupt, SystemExit):
        logging.info("\n[系統] 收到關閉信號 (Ctrl+C)...")
    except Exception as e:
        logging.critical(f"\n[系統] 執行期間發生未預期的嚴重錯誤: {e}", exc_info=True)
    finally:
        if runner:
            runner.shutdown()
        worker_manager.shutdown_workers()
        logging.info("[系統] MoshouSapient 已完全關閉。")