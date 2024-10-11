import cv2
import numpy as np
from ultralytics import YOLO

def resize_frame(frame, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = frame.shape[:2]

    if width is None and height is None:
        return frame

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(frame, dim, interpolation=inter)
    return resized

# 初始化 YOLOv8 模型
racket_model = YOLO("tennis_racket.pt")  # 網球拍子節點
pose_model = YOLO("yolov8n-pose.pt")  # 身體節點

# 讀取影片
video_path = "chu.mp4"  # 請替換為您的影片路徑
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"無法讀取影片：{video_path}")
    exit()

# 獲取影片的屬性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 設置輸出影片的尺寸（原始寬度 + 300像素用於信息面板）
output_width = frame_width + 300
output_height = frame_height

# 創建VideoWriter對象
output_path = "output_labeled_video_with_info.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

# 身體部位與節點編號的對應
body_parts = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 調整幀大小
    frame = resize_frame(frame, width=frame_width)

    # 進行網球拍子節點偵測
    racket_results = racket_model(frame)

    # 進行身體節點偵測
    pose_results = pose_model(frame)

    # 繪製網球拍子節點結果
    annotated_frame = racket_results[0].plot()

    # 創建信息面板
    info_panel = np.zeros((output_height, 300, 3), dtype=np.uint8)

    # 在同一幀上繪製身體節點結果和編號，並更新信息面板
    for result in pose_results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy[0]
            for i, keypoint in enumerate(keypoints):
                x, y = keypoint
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(annotated_frame, str(i), (int(x)+10, int(y)+10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # 更新信息面板
                info_text = f"{i} ({body_parts[i]}): ({x:.2f}, {y:.2f})"
                cv2.putText(info_panel, info_text, (10, 20 + i*20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 合併原始幀和信息面板
    combined_frame = np.hstack((annotated_frame, info_panel))

    # 將處理後的幀寫入輸出影片
    out.write(combined_frame)

    # 顯示處理後的幀
    cv2.imshow("Labeled YOLO Detection with Info", combined_frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"處理完成，輸出影片保存為：{output_path}")