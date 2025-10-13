# src/moshousapient/processors/scene_anomaly_processor.py
"""
定義了 SceneAnomalyProcessor，一個專門負責偵測場景底層異常的處理器。

它不依賴 AI 物件偵測，而是透過分析影像的統計屬性（如亮度、清晰度）
和結構特徵（特徵點匹配）來識別畫面凍結、鏡頭遮擋、失焦或位移等問題。
"""

# 1. 標準庫導入
import logging
import time
from queue import Empty, Queue
from threading import Lock
from typing import Any, Dict, Optional, Tuple

# 2. 第三方庫導入
import cv2
import numpy as np

# 3. 本專案導入
from moshousapient.configs.behavior_config import Config
from moshousapient.processors.base_processor import BaseProcessor
from moshousapient.utils.behavior_analysis_utils import (
    calculate_laplacian_variance,
    calculate_luminance,
)


class SceneAnomalyProcessor(BaseProcessor):
    """
    分析影像流以偵測非 AI 的場景異常。
    """

    def __init__(
        self,
        frame_queue: Queue,
        shared_state: Dict[str, Any],
        state_lock: Lock,
        name: str = "SceneAnomalyProcessor",
    ):
        """
        初始化 SceneAnomalyProcessor。
        """
        super().__init__(name)
        self.frame_queue = frame_queue
        self.shared_state = shared_state
        self.state_lock = state_lock

        # --- 從設定檔載入參數 ---
        self.settings = Config.SCENE_ANOMALY_ALERT_SETTINGS
        self.enabled = self.settings.get("enabled", False)
        self.calibration_duration = self.settings.get("calibration_duration_seconds", 60)
        self.trigger_delay_seconds = self.settings.get("trigger_delay_seconds", 5)

        # --- 亮度檢查參數 ---
        self.lum_check_enabled = self.settings.get("luminance_check", {}).get("enabled", True)
        lum_dev_percent = self.settings.get("luminance_check", {}).get("deviation_threshold_percent", 80)
        self.lum_lower_ratio = float((100 - lum_dev_percent) / 100.0)
        self.lum_upper_ratio = float((100 + lum_dev_percent) / 100.0)

        # --- 清晰度檢查參數 ---
        self.clarity_check_enabled = self.settings.get("clarity_check", {}).get("enabled", True)
        clarity_dev_percent = self.settings.get("clarity_check", {}).get("deviation_threshold_percent", 70)
        self.clarity_lower_ratio = float((100 - clarity_dev_percent) / 100.0)
        self.clarity_absolute_threshold = float(self.settings.get("clarity_check", {}).get("absolute_threshold", 10.0))

        # --- 畫面凍結檢查參數 ---
        self.freeze_check_enabled = self.settings.get("freeze_check", {}).get("enabled", True)
        self.freeze_diff_threshold = float(self.settings.get("freeze_check", {}).get("difference_threshold", 0.01))

        # --- 攝影機篡改檢查參數 ---
        self.tamper_check_enabled = self.settings.get("tamper_check", {}).get("enabled", True)
        self.tamper_match_threshold = float(
            self.settings.get("tamper_check", {}).get("match_threshold_percent", 40) / 100.0
        )

        # --- 內部狀態變數 ---
        self.is_calibrating = True
        self.start_time = 0.0
        self.last_gray_frame: Optional[np.ndarray] = None

        # --- 用於校準的數據收集器 ---
        self._calibration_luminance_values: list[float] = []
        self._calibration_clarity_values: list[float] = []

        # --- 動態基準線 ---
        self.baseline_luminance: float = 0.0
        self.baseline_clarity: float = 0.0

        # --- 篡改檢測參考數據 ---
        # noinspection PyUnresolvedReferences
        self.orb_detector = cv2.ORB_create(nfeatures=500)
        self.reference_keypoints: Optional[Tuple] = None
        self.reference_descriptors: Optional[np.ndarray] = None

        # --- 異常狀態計時器 ---
        self.anomaly_start_time: Dict[str, Optional[float]] = {
            "low_luminance": None,
            "high_luminance": None,
            "low_clarity": None,
            "freeze": None,
            "tamper": None,
        }

    def _update_baselines_and_reference(self, final_calibration_frame: np.ndarray):
        """根據校準數據計算基準線，並設定篡改檢測的參考幀。"""
        if self._calibration_luminance_values:
            self.baseline_luminance = float(np.mean(self._calibration_luminance_values))
        if self._calibration_clarity_values:
            self.baseline_clarity = float(np.mean(self._calibration_clarity_values))

        if self.tamper_check_enabled and final_calibration_frame is not None:
            self.reference_keypoints, self.reference_descriptors = self.orb_detector.detectAndCompute(
                final_calibration_frame, None
            )
            if self.reference_keypoints:
                logging.debug(f"[{self.name}] 已建立篡改檢測參考幀，提取到 {len(self.reference_keypoints)} 個特徵點。")
            else:
                logging.warning(f"[{self.name}] 未能從參考幀中提取到足夠的特徵點，篡改檢測可能不穩定。")

        logging.info(
            f"[{self.name}] 校準完成。動態基準線已建立: "
            f"平均亮度={self.baseline_luminance:.2f}, "
            f"平均清晰度={self.baseline_clarity:.2f}"
        )

    def _check_anomaly_status(self, condition: bool, anomaly_type: str, current_time: float):
        """通用函式，用於更新單個異常狀態的計時器。"""
        if condition:
            if self.anomaly_start_time[anomaly_type] is None:
                self.anomaly_start_time[anomaly_type] = current_time
        else:
            self.anomaly_start_time[anomaly_type] = None

    def _check_anomalies(self, gray_frame: np.ndarray, current_time: float):
        """對單一灰度幀執行所有異常檢查。"""
        # 1. 亮度檢查
        if self.lum_check_enabled:
            luminance = calculate_luminance(gray_frame)
            self._check_anomaly_status(
                luminance < self.baseline_luminance * self.lum_lower_ratio,
                "low_luminance",
                current_time,
            )
            self._check_anomaly_status(
                luminance > self.baseline_luminance * self.lum_upper_ratio,
                "high_luminance",
                current_time,
            )

        # 2. 清晰度檢查
        if self.clarity_check_enabled:
            clarity = calculate_laplacian_variance(gray_frame)
            is_low_clarity = (clarity < self.baseline_clarity * self.clarity_lower_ratio) or (
                clarity < self.clarity_absolute_threshold
            )
            self._check_anomaly_status(is_low_clarity, "low_clarity", current_time)

        # 3. 畫面凍結檢查
        if self.freeze_check_enabled and self.last_gray_frame is not None:
            diff = cv2.absdiff(gray_frame, self.last_gray_frame)
            self._check_anomaly_status(float(np.mean(diff)) < self.freeze_diff_threshold, "freeze", current_time)

        # 4. 攝影機篡改檢查
        if self.tamper_check_enabled and self.reference_descriptors is not None and self.reference_keypoints:
            _, des2 = self.orb_detector.detectAndCompute(gray_frame, None)
            is_tampered = True
            if des2 is not None and len(des2) > 0:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(self.reference_descriptors, des2)
                match_ratio = len(matches) / len(self.reference_keypoints)
                if match_ratio >= self.tamper_match_threshold:
                    is_tampered = False
            self._check_anomaly_status(is_tampered, "tamper", current_time)

    def _update_shared_state(self, current_time: float):
        """檢查異常計時器並更新共享狀態。"""
        active_anomaly: Optional[str] = None
        for anomaly_type, start_time in self.anomaly_start_time.items():
            if start_time and (current_time - start_time) > self.trigger_delay_seconds:
                active_anomaly = anomaly_type
                break

        with self.state_lock:
            if active_anomaly and not self.shared_state.get("scene_anomaly_detected"):
                logging.warning(f"--- [場景異常警報] --- 偵測到: {active_anomaly.replace('_', ' ').upper()}")
            self.shared_state["scene_anomaly_detected"] = active_anomaly is not None
            self.shared_state["scene_anomaly_type"] = active_anomaly

    def _target_func(self):
        """主處理迴圈。"""
        if not self.enabled:
            logging.info(f"[{self.name}] 功能未啟用，處理器將保持閒置。")
            return

        logging.info(f"[{self.name}] 處理器已啟動，正在進行 {self.calibration_duration:.1f} 秒的場景校準...")
        self.start_time = time.time()

        while not self.stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=1)
                current_time = item["time"]

                gray_frame = cv2.cvtColor(item["frame"], cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

                if self.is_calibrating:
                    if self.lum_check_enabled:
                        self._calibration_luminance_values.append(calculate_luminance(gray_frame))
                    if self.clarity_check_enabled:
                        self._calibration_clarity_values.append(calculate_laplacian_variance(gray_frame))

                    if (current_time - self.start_time) >= self.calibration_duration:
                        # 邏輯修正：確保在校準結束時，使用最後一幀作為參考
                        self._update_baselines_and_reference(gray_frame)
                        self.is_calibrating = False
                else:
                    self._check_anomalies(gray_frame, current_time)
                    self._update_shared_state(current_time)

                self.last_gray_frame = gray_frame

            except Empty:
                continue
            except Exception:
                logging.exception(f"[{self.name}] 執行緒發生未預期的錯誤")
                time.sleep(1)

        logging.info(f"[{self.name}] 處理器已停止。")
