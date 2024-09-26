import cv2
from ultralytics import YOLO

# 初始化 YOLOv8 模型
pose_model = YOLO("yolov8n-pose.pt") # 人體節點
# pose_model = YOLO("yolov8n.pt") # 物體偵測(網球)

# 使用 GPU
# pose_model.to('cuda')

# 開啟攝影機
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取攝影機畫面")
        break

    # 進行人體姿態估計
    pose_results = pose_model(frame)
    
    # 繪製人體姿態
    annotated_frame = pose_results[0].plot()

    # 顯示結果
    cv2.imshow("Human Pose Estimation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()