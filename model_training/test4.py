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
BALL_TRAIL_THICKNESS = 4

# 初始化模型
ball_model = YOLO("basketball.pt")

# 讀取影片
video_path = "leftFront.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"無法讀取影片：{video_path}")
    exit()

# 獲取原始影片的FPS並降低播放速度
original_fps = int(cap.get(cv2.CAP_PROP_FPS))
output_fps = original_fps // 3  # 將FPS降為1/3

# 設置輸出影片的尺寸
output_width = FIXED_WIDTH + 400
output_height = FIXED_HEIGHT

# 創建VideoWriter對象
output_path = video_path.replace('.mp4','_ball_trail_slow.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))

# 存儲球的軌跡
ball_trail = []

# 第一次遍歷視頻以收集所有軌跡點
print("第一次遍歷視頻以收集軌跡點...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = resize_frame(frame, width=FIXED_WIDTH, height=FIXED_HEIGHT)
    
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

    # 添加標題區塊
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

    # 合併原始幀和信息面板
    combined_frame = np.hstack((annotated_frame, info_panel))

    # 將處理後的幀寫入輸出影片
    out.write(combined_frame)

    # 顯示處理後的幀
    cv2.imshow("Tennis Ball Detection with Trail", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"處理完成，輸出影片保存為：{output_path}")
print(f"總共處理了 {frame_count} 幀")