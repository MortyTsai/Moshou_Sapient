# export_tensorrt.py
from ultralytics import YOLO


def main():
    """
    使用最佳化參數, 將 YOLO 模型匯出為高效能的 TensorRT 引擎。
    """
    # --- 關鍵修正：指定新的、更強大的模型 ---
    model_name = 'yolo11s.pt'
    engine_name = 'yolo11s.engine'
    # --- 修正結束 ---

    print(f"正在載入 {model_name} 模型...")
    # 確保您已下載 yolo11s.pt 或使用其他 YOLO 模型
    # 如果本地沒有，YOLO() 會自動下載
    model = YOLO(model_name)

    inference_height = 736  # 維持與我們 ANALYSIS_HEIGHT 相同的尺寸
    inference_width = 1280  # 維持與我們 ANALYSIS_WIDTH 相同的尺寸

    print(f"開始以 {inference_height}p 規格將模型匯出為 TensorRT 格式...")
    model.export(
        format='engine',
        device=0,
        half=True,
        imgsz=[inference_height, inference_width],
        workspace=8
    )

    # --- 關鍵修正：提示新的引擎名稱 ---
    print(f"\n模型已成功匯出!")
    print(f"生成的 '{engine_name}' 檔案是為優化後的尺寸而建立的。")
    print(f"請記得更新 config.py 中的 MODEL_PATH！")
    # --- 修正結束 ---


if __name__ == '__main__':
    main()