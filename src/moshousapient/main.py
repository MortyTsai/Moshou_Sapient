# main.py
import logging
import threading
import sys
import torch
import os
from ultralytics import YOLO
import numpy as np
from typing import Optional, Dict, Any
from .config import Config
from .logging_setup import setup_logging
from .database import init_db
from .web.app import create_flask_app
from .components.camera_worker import CameraWorker
from .components.discord_notifier import DiscordNotifier
from .components.runners import RTSPRunner, FileRunner, BaseRunner

def pre_flight_checks():
    """
    執行啟動前的環境檢查。
    """
    logging.info("[系統] 執行啟動前環境檢查...")
    if not torch.cuda.is_available():
        logging.critical("-" * 60)
        logging.critical("[嚴重錯誤] PyTorch 無法偵測到任何可用的 CUDA 設備。")
        logging.critical("請確認:")
        logging.critical("1. NVIDIA 驅動程式已正確安裝。")
        logging.critical("2. 您已安裝支援 GPU 的 PyTorch 版本 (版本號不應包含 '+cpu')。")
        logging.critical("請執行 'pip uninstall torch' 後，參考 PyTorch 官網安裝 GPU 版本。")
        logging.critical("-" * 60)
        return False
    logging.info(f"[系統] CUDA 設備檢查通過。偵測到 GPU: {torch.cuda.get_device_name(0)}")
    return True

def get_camera_config() -> Optional[Dict[str, Any]]:
    """
    根據 .env 設定解析並回傳攝影機設定字典。
    如果設定無效，則回傳 None。
    """

    if Config.VIDEO_SOURCE_TYPE == "FILE":
        logging.info(f"[系統] 影像來源模式: 本地檔案 ({Config.VIDEO_FILE_PATH})")
        if not Config.VIDEO_FILE_PATH or not os.path.exists(Config.VIDEO_FILE_PATH):
            logging.critical(f"[嚴重錯誤] 檔案未找到: {Config.VIDEO_FILE_PATH}。請檢查 .env 設定。")
            return None
        source_uri = Config.VIDEO_FILE_PATH
        source_name = os.path.basename(source_uri)

    elif Config.VIDEO_SOURCE_TYPE == "RTSP":
        logging.info(f"[系統] 影像來源模式: RTSP 即時串流")
        if not Config.RTSP_URL:
            logging.critical("[嚴重錯誤] 未設定完整的 RTSP_URL，請檢查 .env 檔案。")
            return None
        source_uri = Config.RTSP_URL
        source_name = "RTSP-Cam"

    else:
        logging.critical(f"[嚴重錯誤] 無效的 VIDEO_SOURCE_TYPE: '{Config.VIDEO_SOURCE_TYPE}'。請在 .env 中設定為 'RTSP' 或 'FILE'。")
        return None

    return {
        "name": f"Worker-{source_name}",
        "rtsp_url": source_uri,
        "transport_protocol": "udp" if Config.VIDEO_SOURCE_TYPE == "RTSP" else "tcp"
    }

def main():
    # 1. 初始化
    setup_logging()

    if not pre_flight_checks():
        sys.exit(1)

    Config.ensure_captures_dir_exists()
    init_db()

    Config.initialize_dynamic_settings()

    # 2. 載入模型
    logging.info(f"[YOLO] 正在從 {Config.MODEL_PATH} 載入 TensorRT 模型...")
    try:
        model = YOLO(Config.MODEL_PATH, task='detect')
        warmup_frame = np.zeros((Config.ANALYSIS_HEIGHT, Config.ANALYSIS_WIDTH, 3), dtype=np.uint8)
        model.predict(warmup_frame, device=0, verbose=False)
        logging.info("[YOLO] TensorRT 模型已成功載入並預熱。")
        logging.info(f"[Re-ID] 正在載入 {Config.REID_MODEL_PATH} 作為特徵提取器...")
        reid_model = YOLO(Config.REID_MODEL_PATH)
        reid_model.predict(warmup_frame, device=0, verbose=False)
        logging.info("[Re-ID] Re-ID 模型已成功載入並預熱。")

    except Exception as e:
        logging.critical(f"[YOLO] 嚴重錯誤: 無法載入 TensorRT 模型。{e}", exc_info=True)
        return

    # 3. 初始化通知器
    notifier = None
    if Config.DISCORD_ENABLED:
        if Config.DISCORD_TOKEN and Config.DISCORD_CHANNEL_ID:
            notifier = DiscordNotifier(token=Config.DISCORD_TOKEN, channel_id=Config.DISCORD_CHANNEL_ID)
            notifier.start()
        else:
            logging.warning("[系統] Discord 功能已啟用，但未提供完整的憑證。通知功能將被禁用。")
    else:
        logging.info("[系統] Discord 通知功能已被禁用。")

    # 4. 建立攝影機設定與 Worker
    camera_config = get_camera_config()
    if not camera_config:
        if notifier: notifier.stop()
        sys.exit(1)

    workers = [CameraWorker(camera_config, model, reid_model, notifier)]

    # 5. 啟動 Web 儀表板
    logging.info("[系統] 正在背景啟動 Web 儀表板...")
    flask_app = create_flask_app()
    web_thread = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
        daemon=True,
        name="WebDashboardThread"
    )
    web_thread.start()
    logging.info("[系統] Web 儀表板已在 http://127.0.0.1:5000 上運作。")

    # 6. 根據設定選擇並建立執行策略
    runner: Optional[BaseRunner] = None
    if Config.VIDEO_SOURCE_TYPE == "RTSP":
        runner = RTSPRunner(workers, notifier)
    elif Config.VIDEO_SOURCE_TYPE == "FILE":
        runner = FileRunner(workers, notifier)

    # 7. 執行主邏輯
    if runner:
        try:
            runner.run()
        except (KeyboardInterrupt, SystemExit):
            logging.info("\n[系統] 收到關閉信號 (Ctrl+C)...")
        except Exception as e:
            logging.critical(f"\n[系統] 執行期間發生未預期的嚴重錯誤: {e}", exc_info=True)
        finally:
            runner.shutdown()
    else:
        logging.error("[系統] 未能建立有效的執行器，系統即將關閉。")
        if notifier:
            notifier.stop()

if __name__ == "__main__":
    main()