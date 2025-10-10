# scripts/export_tensorrt.py
"""
YOLO 模型轉換為 TensorRT 引擎的輔助腳本。

此腳本用於將標準的 .pt 模型檔案轉換為經過 NVIDIA GPU 優化的高度序列化 TensorRT 引擎。
轉換後的 .engine 檔案可以顯著提升推論速度。

執行方式:
在專案根目錄下運行 `python scripts/export_tensorrt.py`
"""

# 1. 標準庫導入
from pathlib import Path

# 2. 第三方庫導入
from ultralytics import YOLO


def main():
    """
    主執行函式，負責載入 YOLO 模型並將其匯出為 TensorRT 格式。
    """
    # 自動計算專案根目錄，使腳本可以從任何位置安全地執行
    project_root = Path(__file__).resolve().parent.parent

    source_model_path = project_root / "models" / "yolo11s.pt"
    output_engine_path = project_root / "models" / "yolo11s.engine"

    print(f"正在載入來源模型: {source_model_path} ...")

    if not source_model_path.exists():
        print(f"\n[錯誤] 來源模型檔案不存在於 '{source_model_path}'")
        print("請先下載 yolo11s.pt 模型檔案並放置在 models/ 目錄下。")
        return

    try:
        model = YOLO(source_model_path)
    except Exception as e:
        print(f"\n[錯誤] 載入 YOLO 模型時發生錯誤: {e}")
        return

    # 為 MoshouSapient 設定的標準推論尺寸
    inference_height = 736
    inference_width = 1280

    print(f"\n開始以 {inference_height}p ({inference_width}x{inference_height}) 規格將模型匯出為 TensorRT 格式...")
    print("這個過程可能需要幾分鐘時間，具體取決於您的 GPU 性能。")

    try:
        model.export(
            format="engine",
            device=0,
            half=True,
            imgsz=[inference_height, inference_width],
            workspace=8,
        )

        print("\n[成功] 模型已成功匯出!")
        print(f"生成的引擎檔案位於: {output_engine_path}")
        print("MoshouSapient 現在將會自動使用此優化後的模型。")

    except Exception as e:
        print(f"\n[錯誤] 匯出 TensorRT 引擎時發生錯誤: {e}")
        print("請檢查您的 CUDA, cuDNN 和 TensorRT 環境是否已正確安裝並配置。")


if __name__ == "__main__":
    main()
