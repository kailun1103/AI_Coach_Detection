import cv2
import numpy as np
from ultralytics import YOLO

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# 初始化 YOLOv8 模型
racket_model = YOLO("tennis_racket.pt")  # 網球拍子節點
pose_model = YOLO("yolov8n-pose.pt")  # 身體節點

# 讀取圖片
image_path = "123.jpg"  # 請替換為您的圖片路徑
frame = cv2.imread(image_path)

if frame is None:
    print(f"無法讀取圖片：{image_path}")
    exit()

# 調整圖片大小
frame = resize_image(frame, width=800)  # 將圖片寬度調整為800像素，高度等比例縮放

# 進行網球拍子節點偵測
racket_results = racket_model(frame)

# 進行身體節點偵測
pose_results = pose_model(frame)

# 身體部位與節點編號的對應
body_parts = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}

# 繪製網球拍子節點結果
annotated_frame = racket_results[0].plot()

# 在同一幀上繪製身體節點結果和編號
for result in pose_results:
    if result.keypoints is not None:
        keypoints = result.keypoints.xy[0]
        for i, keypoint in enumerate(keypoints):
            x, y = keypoint
            cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(annotated_frame, str(i), (int(x)+10, int(y)+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(f"節點 {i} ({body_parts[i]}): ({x:.2f}, {y:.2f})")

# 顯示結果
cv2.imshow("Labeled YOLO Detection", annotated_frame)
cv2.waitKey(0)  # 等待直到按下任意鍵

# 保存結果
cv2.imwrite("output_labeled_keypoints.jpg", annotated_frame)

cv2.destroyAllWindows()