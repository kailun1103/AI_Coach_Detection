from ultralytics import YOLO

if __name__ == '__main__':
    # 載入模型
    model = YOLO('yolov8n.pt')
    
    # 開始訓練
    results = model.train(
        data='ball/data.yaml',
        epochs=30,
        batch=8,
        imgsz=640,
        device=0 # GPU
    )