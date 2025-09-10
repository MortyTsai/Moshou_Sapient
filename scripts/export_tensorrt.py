# export_tensorrt.py
import os
from ultralytics import YOLO


def main():
    """
    使用最佳化參數，將 YOLO 模型匯出為高效能的 TensorRT 引擎。
    此腳本被設計為可以從專案的任何位置安全地執行。
    """

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_name = os.path.join(PROJECT_ROOT, 'models', 'yolo11s.pt')
    engine_name = os.path.join(PROJECT_ROOT, 'models', 'yolo11s.engine')

    print(f"正在載入來源模型: {model_name} ...")

    if not os.path.exists(model_name):
        print(f"錯誤: 來源模型檔案不存在於 '{model_name}'")
        return

    model = YOLO(model_name)

    inference_height = 736
    inference_width = 1280

    print(f"開始以 {inference_height}p 規格將模型匯出為 TensorRT 格式...")

    model.export(
        format='engine',
        device=0,
        half=True,
        imgsz=[inference_height, inference_width],
        workspace=8
    )

    print(f"\n模型已成功匯出!")
    print(f"生成的引擎檔案位於: {engine_name}")
    print(f"請記得更新 config.py 中的 MODEL_PATH (如果需要)!")


if __name__ == '__main__':
    main()