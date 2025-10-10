# src/moshousapient/core/app_orchestrator.py
"""
MoshouSapient 應用程式的主入口點與協調器。

負責初始化所有組件、根據設定選擇執行策略，並管理應用程式的生命週期。
"""

# 1. 標準庫導入
import logging
import multiprocessing as mp
import sys
import threading
import time
from typing import Any, Dict, Optional

# 2. 第三方庫導入
import torch

# 3. 本專案相對導入
from ..configs.behavior_config import Config
from ..configs.logging_config import configure_logging_for_queue, setup_logging_listener
from ..configs.settings_config import settings
from ..services.database_service import init_db
from ..services.ingestion_service import IngestionService
from ..services.notification_service import NotificationService
from ..services.task_queue_service import TaskQueueService
from ..utils.logging_utils import StreamToLogger
from ..processors.rtsp_processing_pipeline import RTSPPipeline
from ..web.app import create_flask_app
from .producer_runners import BaseRunner, RTSPProducerRunner
from .scheduler import Scheduler
from .worker_manager import WorkerManager


def pre_flight_checks() -> bool:
    """執行應用程式啟動前的基本環境和設定檢查。"""
    logging.debug("[系統] 執行啟動前環境檢查...")
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
        logging.debug(f"[系統] CUDA 設備檢查通過。偵測到 GPU: {torch.cuda.get_device_name(0)}")
    return True


def get_camera_config() -> Optional[Dict[str, Any]]:
    """根據全域設定產生單個攝影機的設定字典。"""
    if Config.VIDEO_SOURCE_TYPE == "RTSP":
        if not Config.RTSP_URL:
            logging.critical("[嚴重錯誤] 未設定完整的 RTSP_URL，請檢查 .env 檔案。")
            return None
        logging.debug("[系統] 影像來源模式: RTSP 即時串流")
        source_uri = Config.RTSP_URL
        source_name = "RTSP-Cam"
        protocol_setting = Config.RTSP_TRANSPORT_PROTOCOL.lower()
        if protocol_setting not in ["udp", "tcp"]:
            logging.warning(f"[設定警告] 無效的 RTSP_TRANSPORT_PROTOCOL: '{protocol_setting}'。將使用預設值 'udp'。")
            transport_protocol = "udp"
        else:
            transport_protocol = protocol_setting
        return {
            "name": f"Pipeline-{source_name}",
            "rtsp_url": source_uri,
            "transport_protocol": transport_protocol,
        }
    return None


def main():
    """應用程式主入口點。"""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    log_queue = mp.Queue(-1)
    configure_logging_for_queue(log_queue)
    logging_listener = setup_logging_listener(log_queue)
    logging_listener.start()

    Config.initialize_static_settings()

    if not pre_flight_checks():
        sys.exit(1)

    init_db()

    # 創建一個跨程序共享的布林標誌，用於指示 RTSP 事件是否正在活躍處理中
    rtsp_event_active_flag = mp.Value("b", False)

    notifier: Optional[NotificationService] = None
    if Config.DISCORD_ENABLED:
        if Config.DISCORD_TOKEN and Config.DISCORD_CHANNEL_ID:
            notifier = NotificationService(token=Config.DISCORD_TOKEN, channel_id=Config.DISCORD_CHANNEL_ID)
            notifier.start()
        else:
            logging.warning("[系統] Discord 功能已啟用，但未提供完整的憑證。通知功能將被禁用。")

    ingestion_service: Optional[IngestionService] = None
    if settings.INGESTION_ENABLED:
        logging.debug("[系統] 檔案攝取服務已啟用，正在初始化...")
        task_queue_for_ingestion = TaskQueueService()
        ingestion_service = IngestionService(
            watch_directory=settings.INGESTION_WATCH_DIR,
            task_queue=task_queue_for_ingestion,
        )
        ingestion_service.start()

    scheduler: Optional[Scheduler] = None
    if settings.SCHEDULER_ENABLED:
        logging.debug("[系統] 智慧排程器已啟用，正在初始化...")
        task_queue_for_scheduler = TaskQueueService()
        # 將共享標誌傳遞給 Scheduler
        scheduler = Scheduler(
            task_queue=task_queue_for_scheduler,
            rtsp_event_active_flag=rtsp_event_active_flag,
        )
        scheduler.start()

    logging.debug("[系統] 正在背景啟動 Web 儀表板...")
    flask_app = create_flask_app()

    def run_flask_silently():
        flask_logger = logging.getLogger("flask_server")
        with StreamToLogger(flask_logger, logging.DEBUG):
            flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    web_thread = threading.Thread(target=run_flask_silently, daemon=True, name="WebDashboardThread")
    web_thread.start()

    runner: Optional[BaseRunner] = None
    if Config.VIDEO_SOURCE_TYPE == "RTSP":
        try:
            from ultralytics import YOLO
            import numpy as np

            logging.debug(f"[YOLO] 正在從 {Config.MODEL_PATH} 載入 TensorRT 模型...")
            model = YOLO(Config.MODEL_PATH, task="detect")
            warmup_frame = np.zeros((Config.ANALYSIS_HEIGHT, Config.ANALYSIS_WIDTH, 3), dtype=np.uint8)
            model.predict(warmup_frame, device=0, verbose=False)
            logging.debug("[YOLO] TensorRT 模型已成功載入並預熱。")

            logging.debug(f"[Re-ID] 正在載入 {Config.REID_MODEL_PATH} 作為特徵提取器...")
            reid_model = YOLO(Config.REID_MODEL_PATH)
            reid_model.embed(warmup_frame, device=0, verbose=False)
            logging.debug("[Re-ID] Re-ID 模型已成功載入並預熱。")

            camera_config = get_camera_config()
            if not camera_config:
                if notifier:
                    notifier.stop()
                sys.exit(1)

            # 將共享標誌傳遞給 RTSP 處理管線
            pipelines = [
                RTSPPipeline(
                    camera_config=camera_config,
                    model=model,
                    reid_model=reid_model,
                    notifier=notifier,
                    rtsp_event_active_flag=rtsp_event_active_flag,
                )
            ]
            runner = RTSPProducerRunner(pipelines, notifier)
        except Exception as e:
            logging.critical(f"[模型載入] 嚴重錯誤: 無法載入 AI 模型。{e}", exc_info=True)
            if notifier:
                notifier.stop()
            sys.exit(1)
    else:
        logging.info("[系統] 未設定 RTSP 模式。系統將在閒置模式下運行，僅監聽背景服務。")

    worker_manager = WorkerManager(num_workers=settings.VIDEO_PROCESSING_WORKERS, log_queue=log_queue)

    logging.info("MoshouSapient 系統啟動完成，Web 儀表板位於 http://127.0.0.1:5000")

    try:
        worker_manager.start_workers()
        if runner:
            runner.run()
        else:
            # 如果沒有 runner (即閒置模式)，主執行緒進入等待模式以保持背景服務運行
            while True:
                time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logging.info("\n[系統] 收到關閉信號，開始優雅關閉...")
    except Exception as e:
        logging.critical(f"\n[系統] 執行期間發生未預期的嚴重錯誤: {e}", exc_info=True)
    finally:
        if runner:
            runner.shutdown()
        worker_manager.shutdown_workers()
        if notifier:
            notifier.stop()
        if ingestion_service:
            ingestion_service.stop()
        if scheduler:
            scheduler.stop()
        logging.info("[系統] MoshouSapient 已完全關閉。")
        logging_listener.stop()


if __name__ == "__main__":
    main()
