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

# 設定固定的輸出大小
FIXED_WIDTH = 1280
FIXED_HEIGHT = 720

# 設定要追蹤的節點編號
TRACKED_KEYPOINTS = [10]
TRAIL_THICKNESS = 6
BALL_TRAIL_THICKNESS = 4

# 初始化模型
ball_model = YOLO('model/tennisball_OD_v1.pt')
pose_model = YOLO("model/yolov8n-pose.pt")

# 讀取影片
video_path = "pro_45_3_1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"無法讀取影片：{video_path}")
    exit()

# 獲取原始影片的FPS並降低播放速度
original_fps = int(cap.get(cv2.CAP_PROP_FPS))
output_fps = original_fps // 5  # 將FPS降為1/3

# 設置輸出影片的尺寸
output_width = FIXED_WIDTH + 400
output_height = FIXED_HEIGHT

# 創建VideoWriter對象
output_path = video_path.replace('.mp4','_full_trail_slow.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))

# 身體部位與節點編號的對應
body_parts = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}

# 創建字典來存儲每個追蹤節點的所有軌跡點
keypoint_trails = {kp: [] for kp in TRACKED_KEYPOINTS}
ball_trail = []  # 存儲球的軌跡

# 第一次遍歷視頻以收集所有軌跡點
print("第一次遍歷視頻以收集軌跡點...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = resize_frame(frame, width=FIXED_WIDTH, height=FIXED_HEIGHT)
    
    # 收集姿態點軌跡
    pose_results = pose_model(frame)
    for result in pose_results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy[0]
            for kp_idx in TRACKED_KEYPOINTS:
                if kp_idx < len(keypoints):
                    x, y = map(int, keypoints[kp_idx])
                    keypoint_trails[kp_idx].append((x, y))
    
    # 收集球的軌跡
    ball_results = ball_model(frame)
    ball_found = False
    for result in ball_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            ball_trail.append((center_x, center_y))
            ball_found = True
            break
        if ball_found:
            break
    if not ball_found:
        ball_trail.append(None)

# 重置視頻捕獲
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_count = 0

print("開始處理視頻並繪製軌跡...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"已處理 {frame_count} 幀")

    frame = resize_frame(frame, width=FIXED_WIDTH, height=FIXED_HEIGHT)
    annotated_frame = frame.copy()

    # 繪製球的軌跡
    current_ball_trail = [p for p in ball_trail[:frame_count] if p is not None]
    for i in range(1, len(current_ball_trail)):
        progress = i / len(current_ball_trail)
        color = (
            0,                          # B
            int(255 * (1 - progress)),  # G
            int(255 * progress)         # R
        )
        cv2.line(annotated_frame, 
                current_ball_trail[i-1],
                current_ball_trail[i], 
                color,
                BALL_TRAIL_THICKNESS)

    # 進行網球檢測
    ball_results = ball_model(frame)
    ball_detected = False
    center_x, center_y = None, None
    conf = 0.0
    
    for result in ball_results:
        boxes = result.boxes
        for box in boxes:
            ball_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            conf = float(box.conf)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
            break
        if ball_detected:
            break

    # 創建信息面板
    info_panel = np.ones((output_height, 400, 3), dtype=np.uint8) * 40

    # 添加標題區塊（深綠色背景）
    title_height = 50
    cv2.rectangle(info_panel, (0, 0), (400, title_height), (0, 100, 0), -1)
    cv2.putText(info_panel, "Tennis Ball Detection", (10, 35), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    # 球的資訊區塊
    y_offset = 70
    if ball_detected:
        cv2.putText(info_panel, "Ball Status: Detected", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_panel, f"Position: ({center_x}, {center_y})", 
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Confidence: {conf:.2f}", 
                    (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    else:
        cv2.putText(info_panel, "Ball Status: Not Detected", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(info_panel, "Position: (None, None)", 
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # 姿態估計標題區塊（棕色背景）
    pose_title_y = y_offset + 90
    cv2.rectangle(info_panel, (0, pose_title_y), (400, pose_title_y + 40), (139, 69, 19), -1)
    cv2.putText(info_panel, "Pose Estimation", (10, pose_title_y + 30), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    # 繪製姿態點資訊
    pose_y_offset = pose_title_y + 60
    pose_results = pose_model(frame)
    
    if len(pose_results) > 0 and pose_results[0].keypoints is not None:
        keypoints = pose_results[0].keypoints.xy[0]
        
        # 標記姿態點
        for i, keypoint in enumerate(keypoints):
            x, y = map(int, keypoint)
            color = (0, 0, 255) if i in TRACKED_KEYPOINTS else (0, 255, 0)
            cv2.circle(annotated_frame, (x, y), 5, color, -1)
            cv2.putText(annotated_frame, str(i), (x+10, y+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 顯示座標資訊（單列）
            part_name = body_parts.get(i, "unknown")
            info_text = f"{part_name:15s}: ({x:4d}, {y:4d})"
            cv2.putText(info_panel, info_text, 
                    (10, pose_y_offset + i*25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(info_panel, "No keypoints detected", (10, pose_y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # 繪製身體節點軌跡
    for kp_idx, trail in keypoint_trails.items():
        current_trail = trail[:frame_count]
        for i in range(1, len(current_trail)):
            progress = i / len(current_trail)
            color = (
                int(255 * (1 - progress)),
                int(255 * progress),
                0
            )
            cv2.line(annotated_frame, 
                    current_trail[i-1],
                    current_trail[i], 
                    color,
                    TRAIL_THICKNESS)

    # 合併原始幀和信息面板
    combined_frame = np.hstack((annotated_frame, info_panel))

    # 將處理後的幀寫入輸出影片
    out.write(combined_frame)

    # 顯示處理後的幀
    cv2.imshow("Tennis Ball and Pose Detection with Full Trail", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"處理完成，輸出影片保存為：{output_path}")
print(f"總共處理了 {frame_count} 幀")