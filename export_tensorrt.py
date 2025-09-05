# export_tensorrt.py
from ultralytics import YOLO

def main():
    """
    使用最佳化參數, 將 YOLO 模型匯出為高效能的 TensorRT 引擎。
    """
    print("正在載入 YOLOv11n.pt 模型...")
    # 確保您已下載 yolov11n.pt 或使用其他 YOLO 模型
    model = YOLO('yolov11n.pt')

    inference_height = 720
    inference_width = 1280

    print(f"開始以 {inference_height}p 規格將模型匯出為 TensorRT 格式...")
    model.export(
        format='engine',
        device=0,
        half=True,
        imgsz=[inference_height, inference_width],
        workspace=8
    )
    print("\n模型已成功匯出!")
    print(f"生成的 'yolo11n.engine' 檔案是為優化後的尺寸而建立的。")

if __name__ == '__main__':
    main()