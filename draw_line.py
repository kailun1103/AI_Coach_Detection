import cv2
import numpy as np
import json

# 讀取 JSON 數據
with open('answer.json', 'r') as file:
    data = json.load(file)



def scale_coordinates(data, scale_factor):
    # 找出所有 x 和 y 坐標的最小值和最大值
    min_x = min(int(point['x']) for point in data)
    max_x = max(int(point['x']) for point in data)
    min_y = min(int(point['y']) for point in data)
    max_y = max(int(point['y']) for point in data)

    # 計算中心點
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # 對每個點進行縮放
    for point in data:
        x = int(point['x'])
        y = int(point['y'])
        
        # 相對於中心點進行縮放
        new_x = center_x + (x - center_x) * scale_factor
        new_y = center_y + (y - center_y) * scale_factor
        
        # 更新坐標
        point['x'] = str(int(new_x))
        point['y'] = str(int(new_y))
    return data

scaled_data = scale_coordinates(data, 0.5762)

with open('scaled_answer.json', 'w') as file:
    json.dump(scaled_data, file, indent=2)


for i in data:
    i['x'] = str(int(i['x'])-535)
    i['y'] = str(int(i['y'])-55)


# 定義輸入和輸出視頻路徑
input_video_path = 'chu.mp4'
output_video_path = 'output_trajectory_robust.mp4'

# 打開輸入視頻
cap = cv2.VideoCapture(input_video_path)

# 獲取視頻屬性
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 創建 VideoWriter 對象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 定義軌跡繪製參數
trail_color = (0, 255, 255)  # 黃色
trail_thickness = 5
point_color = (0, 0, 255)    # 紅色
point_radius = 7

trajectory_points = []
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 安全地獲取當前幀的點
    if frame_number < len(data):
        current_data = data[frame_number]
        current_x = current_data.get('x')
        current_y = current_data.get('y')
        
        if current_x is not None and current_y is not None:
            current_point = (int(current_x), int(current_y))
            trajectory_points.append(current_point)

    # 繪製軌跡
    if len(trajectory_points) > 1:
        cv2.polylines(frame, [np.array(trajectory_points)], False, trail_color, trail_thickness)

    # 在當前點繪製圓圈
    if trajectory_points:
        cv2.circle(frame, trajectory_points[-1], point_radius, point_color, -1)

    # 寫入幀
    out.write(frame)

    frame_number += 1
    if frame_number % 10 == 0:
        print(f"處理幀 {frame_number}")

# 釋放資源
cap.release()
out.release()

print(f"視頻處理完成。輸出視頻保存至 {output_video_path}")
print(f"總處理幀數：{frame_number}")
print(f"軌跡數據點數：{len(data)}")