# main.py
import time
import logging
import threading
import sys
import torch
from ultralytics import YOLO
import numpy as np

from config import Config
from logging_setup import setup_logging
from components.camera_worker import CameraWorker
from components.discord_notifier import DiscordNotifier
from database import init_db
from web_dashboard import create_flask_app


def pre_flight_checks():
    """
    執行啟動前的環境檢查。
    """
    logging.info("[系統] 執行啟動前環境檢查...")
    if not torch.cuda.is_available():
        logging.critical("-" * 60)
        logging.critical("[嚴重錯誤] PyTorch 無法偵測到任何可用的 CUDA 設備。")
        logging.critical("請確認：")
        logging.critical("1. NVIDIA 驅動程式已正確安裝。")
        logging.critical("2. 您已安裝支援 GPU 的 PyTorch 版本 (版本號不應包含 '+cpu')。")
        logging.critical("請執行 'pip uninstall torch' 後，參考 PyTorch 官網安裝 GPU 版本。")
        logging.critical("-" * 60)
        return False
    logging.info(f"[系統] CUDA 設備檢查通過。偵測到 GPU: {torch.cuda.get_device_name(0)}")
    return True


def main():
    # 1. 初始化
    setup_logging()

    if not pre_flight_checks():
        sys.exit(1)

    Config.ensure_captures_dir_exists()
    init_db()

    # 2. 載入模型
    logging.info(f"[YOLO] 正在從 {Config.MODEL_PATH} 載入 TensorRT 模型...")
    try:
        model = YOLO(Config.MODEL_PATH, task='detect')
        warmup_frame = np.zeros((Config.ANALYSIS_HEIGHT, Config.ANALYSIS_WIDTH, 3), dtype=np.uint8)
        model.predict(warmup_frame, device=0, verbose=False)
        logging.info("[YOLO] TensorRT 模型已成功載入並預熱。")

        logging.info("[Re-ID] 正在載入 yolo11s-cls.pt 作為特徵提取器...")
        reid_model = YOLO('yolo11s-cls.pt')
        # 預熱 Re-ID 模型
        reid_model.predict(warmup_frame, device=0, verbose=False)
        logging.info("[Re-ID] Re-ID 模型已成功載入並預熱。")

    except Exception as e:
        logging.critical(f"[YOLO] 嚴重錯誤: 無法載入 TensorRT 模型。{e}", exc_info=True)
        return

    # 3. 初始化通知器
    notifier = None
    if Config.DISCORD_TOKEN and Config.DISCORD_CHANNEL_ID:
        notifier = DiscordNotifier(token=Config.DISCORD_TOKEN, channel_id=Config.DISCORD_CHANNEL_ID)
        notifier.start()
    else:
        logging.warning("[系統] 未設定 Discord 憑證, 通知功能將被禁用。")

    # 4. 建立攝影機設定與 Worker
    if not Config.RTSP_URL_HIGH_RES:
        logging.critical("[嚴重錯誤] 未設定 Tapo 攝影機憑證, 請檢查 .env 檔案。")
        if notifier: notifier.stop()
        return

    camera_configs = [
        {"name": "Front-Door-Cam", "rtsp_url": Config.RTSP_URL_HIGH_RES, "transport_protocol": "udp"},
    ]
    workers = [CameraWorker(config, model, reid_model, notifier) for config in camera_configs]

    # 5. 啟動 Web 儀表板 (在背景執行緒)
    logging.info("[系統] 正在背景啟動 Web 儀表板...")
    flask_app = create_flask_app()
    web_thread = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
        daemon=True,
        name="WebDashboardThread"
    )
    web_thread.start()
    logging.info("[系統] Web 儀表板已在 http://127.0.0.1:5000 上運作。")

    # 6. 主迴圈: 啟動並監控
    stop_main_event = threading.Event()
    try:
        for worker in workers:
            worker.start()
        while not stop_main_event.is_set():
            time.sleep(Config.HEALTH_CHECK_INTERVAL)
            for worker in workers:
                if not worker.is_alive():
                    logging.critical(f"[{worker.name}] 的一個或多個核心執行緒已停止運作! 系統將準備關閉。")
                    stop_main_event.set()
                    break
    except (KeyboardInterrupt, SystemExit):
        logging.info("\n[系統] 收到關閉信號 (Ctrl+C)...")
    except Exception as e:
        logging.critical(f"\n[系統] 發生未預期的嚴重錯誤: {e}", exc_info=True)
    finally:
        logging.info("\n[系統] 正在優雅地關閉所有服務, 請稍候...")
        for worker in workers:
            worker.stop()
        if notifier:
            notifier.stop()
        logging.info("[系統] 系統已安全關閉。")


if __name__ == "__main__":
    main()