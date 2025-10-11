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
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, Dict, Optional, Tuple

# 2. 第三方庫導入
import numpy as np
import torch
from ultralytics import YOLO

# 3. 本專案導入
from moshousapient.configs.behavior_config import Config
from moshousapient.configs.logging_config import configure_logging_for_queue, setup_logging_listener
from moshousapient.configs.settings_config import settings
from moshousapient.core.producer_runners import BaseRunner, RTSPProducerRunner
from moshousapient.core.scheduler import Scheduler
from moshousapient.core.worker_manager import WorkerManager
from moshousapient.processors.rtsp_processing_pipeline import RTSPPipeline
from moshousapient.services.database_service import init_db
from moshousapient.services.ingestion_service import IngestionService
from moshousapient.services.notification_service import NotificationService
from moshousapient.services.task_queue_service import TaskQueueService
from moshousapient.utils.logging_utils import StreamToLogger
from moshousapient.web.app import create_flask_app


def _setup_logging() -> Tuple[mp.Queue, Any]:
    """設定中央化日誌系統。"""
    log_queue = mp.Queue(-1)
    configure_logging_for_queue(log_queue)
    logging_listener = setup_logging_listener(log_queue)
    logging_listener.start()
    return log_queue, logging_listener


def _pre_flight_checks() -> bool:
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


def _initialize_services(
    rtsp_event_active_flag: MpEvent,
) -> Tuple[Optional[NotificationService], Optional[IngestionService], Optional[Scheduler]]:
    """初始化所有背景服務。"""
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
        scheduler = Scheduler(task_queue=task_queue_for_scheduler, rtsp_event_active_flag=rtsp_event_active_flag)
        scheduler.start()

    return notifier, ingestion_service, scheduler


def _start_web_dashboard():
    """在背景執行緒中啟動 Flask Web 儀表板。"""
    logging.debug("[系統] 正在背景啟動 Web 儀表板...")
    flask_app = create_flask_app()

    def run_flask_silently():
        flask_logger = logging.getLogger("flask_server")
        with StreamToLogger(flask_logger, logging.DEBUG):
            flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    web_thread = threading.Thread(target=run_flask_silently, daemon=True, name="WebDashboardThread")
    web_thread.start()


def _get_camera_config() -> Optional[Dict[str, Any]]:
    """根據全域設定產生單個攝影機的設定字典。"""
    if not Config.RTSP_URL:
        logging.critical("[嚴重錯誤] 未設定完整的 RTSP_URL，請檢查 .env 檔案。")
        return None

    logging.debug("[系統] 影像來源模式: RTSP 即時串流")
    protocol_setting = Config.RTSP_TRANSPORT_PROTOCOL.lower()
    if protocol_setting not in ["udp", "tcp"]:
        logging.warning(f"[設定警告] 無效的 RTSP_TRANSPORT_PROTOCOL: '{protocol_setting}'。將使用預設值 'udp'。")
        transport_protocol = "udp"
    else:
        transport_protocol = protocol_setting

    return {
        "name": "Pipeline-RTSP-Cam",
        "rtsp_url": Config.RTSP_URL,
        "transport_protocol": transport_protocol,
    }


def _initialize_rtsp_runner(
    notifier: Optional[NotificationService], rtsp_event_active_flag: MpEvent
) -> Optional[RTSPProducerRunner]:
    """載入 AI 模型並初始化 RTSP 模式的 Runner。"""
    if Config.VIDEO_SOURCE_TYPE != "RTSP":
        logging.info("[系統] 未設定 RTSP 模式。系統將在閒置模式下運行，僅監聽背景服務。")
        return None

    try:
        logging.debug(f"[YOLO] 正在從 {Config.MODEL_PATH} 載入 TensorRT 模型...")
        model = YOLO(Config.MODEL_PATH, task="detect")
        warmup_frame = np.zeros((Config.ANALYSIS_HEIGHT, Config.ANALYSIS_WIDTH, 3), dtype=np.uint8)
        model.predict(warmup_frame, device=0, verbose=False)
        logging.debug("[YOLO] TensorRT 模型已成功載入並預熱。")

        logging.debug(f"[Re-ID] 正在載入 {Config.REID_MODEL_PATH} 作為特徵提取器...")
        reid_model = YOLO(Config.REID_MODEL_PATH)
        reid_model.embed(warmup_frame, device=0, verbose=False)
        logging.debug("[Re-ID] Re-ID 模型已成功載入並預熱。")

        camera_config = _get_camera_config()
        if not camera_config:
            return None

        pipelines = [
            RTSPPipeline(
                camera_config=camera_config,
                model=model,
                reid_model=reid_model,
                notifier=notifier,
                rtsp_event_active_flag=rtsp_event_active_flag,
            )
        ]
        return RTSPProducerRunner(pipelines, notifier)
    except Exception:
        logging.critical("[模型載入] 嚴重錯誤: 無法載入 AI 模型。", exc_info=True)
        return None


def _run_app_lifecycle(
    runner: Optional[BaseRunner],
    worker_manager: WorkerManager,
    services: Tuple[Optional[NotificationService], Optional[IngestionService], Optional[Scheduler]],
    logging_listener: Any,
):
    """執行應用程式的主生命週期，包括啟動、運行和關閉。"""
    notifier, ingestion_service, scheduler = services
    try:
        worker_manager.start_workers()
        if runner:
            runner.run()
        else:
            while True:
                time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logging.info("\n[系統] 收到關閉信號，開始優雅關閉...")
    except Exception:
        logging.critical("\n[系統] 執行期間發生未預期的嚴重錯誤", exc_info=True)
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


def main():
    """應用程式主入口點。"""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    log_queue, logging_listener = _setup_logging()
    Config.initialize_static_settings()

    if not _pre_flight_checks():
        sys.exit(1)

    init_db()
    _start_web_dashboard()

    rtsp_event_active_flag = mp.Value("b", False)
    services = _initialize_services(rtsp_event_active_flag)
    notifier, _, _ = services

    runner = _initialize_rtsp_runner(notifier, rtsp_event_active_flag)
    if Config.VIDEO_SOURCE_TYPE == "RTSP" and runner is None:
        # 如果 RTSP 模式初始化失敗，則優雅關閉
        _, ingestion_service, scheduler = services
        if notifier:
            notifier.stop()
        if ingestion_service:
            ingestion_service.stop()
        if scheduler:
            scheduler.stop()
        logging_listener.stop()
        sys.exit(1)

    worker_manager = WorkerManager(num_workers=settings.VIDEO_PROCESSING_WORKERS, log_queue=log_queue)
    logging.info("MoshouSapient 系統啟動完成，Web 儀表板位於 http://127.0.0.1:5000")

    _run_app_lifecycle(runner, worker_manager, services, logging_listener)


if __name__ == "__main__":
    main()
