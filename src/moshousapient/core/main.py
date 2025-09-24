# src/moshousapient/core/main.py

import logging
import threading
import sys
import torch
from typing import Optional, Dict, Any

from ..config import Config
from ..logging_setup import setup_logging
from ..database import init_db
from ..web.app import create_flask_app
from .camera_worker import CameraWorker
from ..services.discord_notifier import DiscordNotifier
from .runners import RTSPRunner, FileRunner, BaseRunner


def pre_flight_checks() -> bool:
    """執行啟動前的環境檢查。"""
    logging.info("[系統] 執行啟動前環境檢查...")
    # 只有在需要 GPU 的模式下才檢查 CUDA
    if Config.VIDEO_SOURCE_TYPE == "RTSP":
        if not torch.cuda.is_available():
            logging.critical("-" * 60)
            logging.critical("[嚴重錯誤] PyTorch 無法偵測到任何可用的 CUDA 設備。")
            logging.critical("請確認:")
            logging.critical("  1. NVIDIA 驅動程式已正確安裝。")
            logging.critical("  2. 您已安裝支援 GPU 的 PyTorch 版本 (版本號不應包含 '+cpu')。")
            logging.critical("  請執行 'pip uninstall torch' 後, 參考 PyTorch 官網安裝 GPU 版本。")
            logging.critical("-" * 60)
            return False
        logging.info(f"[系統] CUDA 設備檢查通過。偵測到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("[系統] 在 FILE 模式下，跳過 CUDA 設備檢查。")
    return True


def get_camera_config() -> Optional[Dict[str, Any]]:
    """根據 .env 設定解析並回傳攝影機設定字典。"""
    if Config.VIDEO_SOURCE_TYPE == "RTSP":
        if not Config.RTSP_URL:
            logging.critical("[嚴重錯誤] 未設定完整的 RTSP_URL, 請檢查 .env 檔案。")
            return None
        logging.info(f"[系統] 影像來源模式: RTSP 即時串流")
        source_uri = Config.RTSP_URL
        source_name = "RTSP-Cam"
        protocol_setting = Config.RTSP_TRANSPORT_PROTOCOL.lower()
        if protocol_setting not in ["udp", "tcp"]:
            logging.warning(f"[設定警告] 無效的 RTSP_TRANSPORT_PROTOCOL: '{protocol_setting}'。將使用預設值 'udp'。")
            transport_protocol = "udp"
        else:
            transport_protocol = protocol_setting

        return {
            "name": f"Worker-{source_name}",
            "rtsp_url": source_uri,
            "transport_protocol": transport_protocol
        }
    return None


def main():
    # 1. 基礎初始化
    setup_logging()
    Config.initialize_static_settings()

    if not pre_flight_checks():
        sys.exit(1)

    init_db()
    Config.initialize_dynamic_settings()

    # 2. 初始化通知器 (所有模式共用)
    notifier = None
    if Config.DISCORD_ENABLED:
        if Config.DISCORD_TOKEN and Config.DISCORD_CHANNEL_ID:
            notifier = DiscordNotifier(token=Config.DISCORD_TOKEN, channel_id=Config.DISCORD_CHANNEL_ID)
            notifier.start()
        else:
            logging.warning("[系統] Discord 功能已啟用, 但未提供完整的憑證。通知功能將被禁用。")
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

    # 4. 根據設定選擇並建立執行策略
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
            reid_model.predict(warmup_frame, device=0, verbose=False)
            logging.info("[Re-ID] Re-ID 模型已成功載入並預熱。")

            camera_config = get_camera_config()
            if not camera_config:
                if notifier: notifier.stop()
                sys.exit(1)

            workers = [CameraWorker(camera_config, model, reid_model, notifier)]
            runner = RTSPRunner(workers, notifier)

        except Exception as e:
            logging.critical(f"[模型載入] 嚴重錯誤: 無法載入 AI 模型。{e}", exc_info=True)
            if notifier: notifier.stop()
            sys.exit(1)

    elif Config.VIDEO_SOURCE_TYPE == "FILE":
        runner = FileRunner(workers=[], notifier=notifier)

    else:
        logging.critical(
            f"[嚴重錯誤] 無效的 VIDEO_SOURCE_TYPE: '{Config.VIDEO_SOURCE_TYPE}'。請在 .env 中設定為 'RTSP' 或 'FILE'。")
        if notifier: notifier.stop()
        sys.exit(1)

    # 5. 執行主邏輯
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
        logging.error("[系統] 未能建立有效的執行器, 系統即將關閉。")
        if notifier:
            notifier.stop()


if __name__ == "__main__":
    main()