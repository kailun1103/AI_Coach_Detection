import numpy as np
from ultralytics import YOLO
import cv2 # OpenCV
import json

# Load YOLOv8 Pose model
model = YOLO('yolov8n-pose.pt')

# Define input video path and JSON output path
input_video_path = 'answer.mp4'
output_video_path = f'{input_video_path.replace('.mp4','')}_trajectory.mp4'
json_output_path = f'{input_video_path.replace('.mp4','')}_trajectory.json'

# Open the video
cap = cv2.VideoCapture(input_video_path) # cv2.VideoCapture()可以捕捉影片詳細資訊

# Get video information
fps = int(cap.get(cv2.CAP_PROP_FPS)) # 幀率（Frames Per Second，FPS）
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 幀寬度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #　幀高度

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # cv2.VideoWriter_fourcc用於指定影片編碼器，'mp4v' 是 MPEG-4 編碼的四字符代碼/*'mp4v' 將字符串 'mp4v' 拆分為單個字符。
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height)) # out 用於寫入處理後的影片幀。



# 開始處理影片
frame_number = 0
trajectory_points = []
last_point = None
min_speed = 1  # 最小速度閾值
smoothing_factor = 0.3  # 平滑因子
trail_thickness = 5  # 軌跡厚度

def smooth_point(last, current, factor): # 用於平滑軌跡點，減少抖動。
    if last is None:
        return current
    return tuple(int(last[i] * (1 - factor) + current[i] * factor) for i in range(2))

# List to store JSON data
json_data = []

# 開始主循環處理影片
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: # 結束處理影片會break
        break 

    # 對每一幀執行姿態檢測
    results = model(frame)

    # 初始化幀數據
    frame_data = {
        "frame": frame_number,
        "x": None,
        "y": None,
        "vector_x": None,
        "vector_y": None,
        "right_shoulder": {"x": None, "y": None},
        "right_hip": {"x": None, "y": None}
    }
    # 處理檢測結果
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy[0].cpu().numpy()

            # 提取右手腕的位置（關鍵點 10）
            if keypoints.shape[0] > 10:
                right_wrist = tuple(map(int, keypoints[10][:2]))  # 右手腕 (關鍵點 10)
                right_shoulder = tuple(map(int, keypoints[6][:2]))  # 右肩 (關鍵點 6)
                right_hip = tuple(map(int, keypoints[12][:2]))  # 右髖 (關鍵點 12)

                # 平滑和記錄軌跡點
                smoothed_point = smooth_point(last_point, right_wrist, smoothing_factor)
                frame_data["x"], frame_data["y"] = int(smoothed_point[0]), int(smoothed_point[1])
                frame_data["right_shoulder"]["x"], frame_data["right_shoulder"]["y"] = right_shoulder
                frame_data["right_hip"]["x"], frame_data["right_hip"]["y"] = right_hip

                # 計算運動向量
                if last_point:
                    vector = (smoothed_point[0] - last_point[0], smoothed_point[1] - last_point[1])
                    frame_data["vector_x"], frame_data["vector_y"] = vector

                # 更新軌跡
                if last_point is None or np.linalg.norm(np.array(smoothed_point) - np.array(last_point)) >= min_speed:
                    trajectory_points.append(smoothed_point)
                    last_point = smoothed_point

    json_data.append(frame_data)
    # print(frame_data)

    frame_number += 1

    # 記錄數據並更新進度
    if frame_number % 10 == 0:
        print(f"Processing frame {frame_number}")

# Release resources
cap.release()


# print(json_data)

def find_min_values(data):
    min_x = float('inf')
    min_y = float('inf')
    min_x_frames = []
    min_y_frames = []
    
    for item in data:
        if item['x'] is not None and item['y'] is not None:
            # Handle x
            if item['x'] < min_x:
                min_x = item['x']
                min_x_frames = [item['frame']]
            elif item['x'] == min_x:
                min_x_frames.append(item['frame'])
            
            # Handle y
            if item['y'] < min_y:
                min_y = item['y']
                min_y_frames = [item['frame']]
            elif item['y'] == min_y:
                min_y_frames.append(item['frame'])
    
    return min(min_x_frames), max(min_y_frames)

# 判斷開始揮拍和節數揮拍的偵數
start_frame, end_frame = find_min_values(json_data)

# 把開始揮拍到節數揮拍的向量數據寫進json裡面
filtered_json_data = [frame_data for frame_data in json_data if start_frame <= frame_data['frame'] <= end_frame]
with open(json_output_path, 'w') as f:
    json.dump(filtered_json_data, f, indent=2)


with open('answer.json', 'r') as file:
    answer_json_data = json.load(file)


# 重新打開影片，並繪製軌跡
cap = cv2.VideoCapture(input_video_path)

# 初始化幀計數器和軌跡點列表。
frame_number = 0
trajectory_points = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 檢查當前幀是否在我們感興趣的範圍內。
    if start_frame <= frame_number <= end_frame:
        # 從過濾後的 JSON 數據中獲取當前幀的點坐標。
        point = (filtered_json_data[frame_number - start_frame]['x'], filtered_json_data[frame_number - start_frame]['y'])
        # 如果坐標有效，將其添加到軌跡點列表中。
        if point[0] is not None and point[1] is not None:
            trajectory_points.append(point)

    # 如果有多於一個點，在幀上繪製軌跡線。
    if len(trajectory_points) > 1:
        cv2.polylines(frame, [np.array(trajectory_points)], False, (0, 255, 255), trail_thickness)

    # 在當前點上繪製一個圓圈，標記最新的位置。
    if frame_number <= end_frame and trajectory_points:
        cv2.circle(frame, trajectory_points[-1], trail_thickness + 2, (0, 0, 255), -1)

    # 將處理後的幀寫入輸出影片
    out.write(frame)

    frame_number += 1

    # 每處理 10 幀就顯示一次進度。
    if frame_number % 10 == 0:
        print(f"Drawing trajectory: Processing frame {frame_number}")

# Release resources
cap.release()
out.release()

print(f"Video processing complete.")
print(f"JSON data saved to {json_output_path}")
print(f"Output video with trajectory saved to {output_video_path}")
print(f"Total frames processed: {frame_number}")
print(f"Trajectory drawn from frame {start_frame} to frame {end_frame}")