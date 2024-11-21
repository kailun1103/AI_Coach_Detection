import numpy as np
from ultralytics import YOLO
import cv2
import os
import json

# 載入 YOLOv8 Pose 模型
model = YOLO('yolov8n-pose.pt')

# 定義輸入影片路徑、輸出影片路徑和JSON輸出路徑
video_path = 'E:/git_repos/AI_Coach_Detection/tennis_full_video/test3.mp4'
output_video_path = 'output.mp4'
json_output_path = 'trajectory_data.json'

# 打開影片
cap = cv2.VideoCapture(video_path)

# 獲取影片的基本資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 創建VideoWriter對象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_number = 0
trajectory_points = []
last_point = None
min_speed = 1  # 最小速度閾值
smoothing_factor = 0.3  # 平滑因子
trail_thickness = 5  # 增加軌跡線的粗細

def smooth_point(last, current, factor):
    if last is None:
        return current
    return tuple(int(last[i] * (1 - factor) + current[i] * factor) for i in range(2))

# 創建一個蒙版來繪製軌跡
mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# 用於存儲JSON數據的列表
json_data = []

# 逐針處理影片
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # 如果讀取失敗或到影片結尾，停止迴圈
    
    # 在每一針上執行姿態偵測
    results = model(frame)
    
    frame_data = {"frame": frame_number, "x": None, "y": None, "vector_x": None, "vector_y": None} # vector_x、vector_y相對於前一幀的位移
    
    # 遍歷偵測結果
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy[0].cpu().numpy()
            
            # 右手腕是關鍵點10
            if keypoints.shape[0] > 10:
                right_wrist = tuple(map(int, keypoints[10][:2]))  # 轉換為整數元組
                
                # 平滑處理
                smoothed_point = smooth_point(last_point, right_wrist, smoothing_factor)
                
                frame_data["x"], frame_data["y"] = smoothed_point
                
                if last_point:
                    vector = (smoothed_point[0] - last_point[0], smoothed_point[1] - last_point[1])
                    frame_data["vector_x"], frame_data["vector_y"] = vector
                
                if last_point is None or np.linalg.norm(np.array(smoothed_point) - np.array(last_point)) >= min_speed:
                    trajectory_points.append(smoothed_point)
                    last_point = smoothed_point
    
    json_data.append(frame_data)
    
    # 繪製軌跡到蒙版上
    if len(trajectory_points) > 1:
        # 創建一個臨時蒙版來繪製新的軌跡段
        temp_mask = np.zeros_like(mask)
        cv2.polylines(temp_mask, [np.array(trajectory_points)], False, (0, 255, 255), trail_thickness)
        
        # 將新的軌跡段添加到主蒙版上，並使舊的軌跡逐漸變淡
        mask = cv2.addWeighted(mask, 0.7, temp_mask, 1, 0)
    
    # 將軌跡蒙版疊加到原始幀上
    frame_with_trails = cv2.addWeighted(frame, 1, mask, 0.8, 0)
    
    # 在最新的點上繪製一個醒目的點
    if trajectory_points:
        cv2.circle(frame_with_trails, trajectory_points[-1], trail_thickness + 2, (0, 0, 255), -1)
    
    # 添加幀號到影像
    cv2.putText(frame_with_trails, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 將處理後的影像寫入輸出影片
    out.write(frame_with_trails)
    
    frame_number += 1
    
    # 顯示進度
    if frame_number % 10 == 0:
        print(f"Processing frame {frame_number}")

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

# 將數據寫入JSON文件
with open(json_output_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"Video processing complete. Output saved to {output_video_path}")
print(f"JSON data saved to {json_output_path}")
print(f"Total frames processed: {frame_number}")
print(f"Total trajectory points: {len(trajectory_points)}")