import numpy as np
from ultralytics import YOLO
import math
import cv2
import json

def save_json_to_jsonfile(json_data, json_path):
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

def get_json_from_jsonfile(answer_json_path):
    with open(answer_json_path, 'r') as file:
        answer_json_data = json.load(file)
    return answer_json_data

def scale_coordinates(answer_json_data, scale_ratio):
    # 找出所有 x 和 y 坐標的最小值和最大值
    min_x = min(int(point["right_wrist"]['x']) for point in answer_json_data)
    max_x = max(int(point["right_wrist"]['x']) for point in answer_json_data)
    min_y = min(int(point["right_wrist"]['y']) for point in answer_json_data)
    max_y = max(int(point["right_wrist"]['y']) for point in answer_json_data)

    # 計算中心點
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # 對每個點進行縮放
    for point in answer_json_data:
        x = int(point["right_wrist"]['x'])
        y = int(point["right_wrist"]['y'])
        
        # 相對於中心點進行縮放
        new_x = center_x + (x - center_x) * scale_ratio
        new_y = center_y + (y - center_y) * scale_ratio
        
        # 更新坐標
        point["right_wrist"]['x'] = int(new_x)
        point["right_wrist"]['y'] = int(new_y)
    return answer_json_data

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def smooth_point(last, current, factor): # 用於平滑軌跡點，減少抖動。
    if last is None:
        return current
    return tuple(int(last[i] * (1 - factor) + current[i] * factor) for i in range(2))


def determine_swing_trajectory(data):
    min_wrist_x = float('inf')
    min_wrist_y = float('inf')
    min_ball_x = float('inf')  # Changed from ball_x to min_ball_x
    min_wrist_x_frames = []
    min_wrist_y_frames = []
    ball_return = None

    for item in data:
        # Handle wrist position
        if item["right_wrist"]['x'] is not None and item["right_wrist"]['y'] is not None:
            # Handle wrist x
            if item["right_wrist"]['x'] < min_wrist_x:
                min_wrist_x = item["right_wrist"]['x']
                min_wrist_x_frames = [item['frame']]
            elif item["right_wrist"]['x'] == min_wrist_x:
                min_wrist_x_frames.append(item['frame'])
            # Handle wrist y
            if item["right_wrist"]['y'] < min_wrist_y:
                min_wrist_y = item["right_wrist"]['y']
                min_wrist_y_frames = [item['frame']]
            elif item["right_wrist"]['y'] == min_wrist_y:
                min_wrist_y_frames.append(item['frame'])
        
        # Handle tennis ball position
        if item["tennis_ball"]['x'] is not None:
            if item["tennis_ball"]['x'] < min_ball_x:  # Changed from ball_x to min_ball_x
                min_ball_x = item["tennis_ball"]['x']
                ball_return = item['frame']

    swing_start = min(min_wrist_x_frames)
    swing_end = max(min_wrist_y_frames)
    
    return swing_start, swing_end, ball_return


def process_video(yolo_pose_model, tennis_ball_model, input_video_path, output_video_path, output_json_path):
    
    # -------------step1: 處理影片的每一偵資訊-------------

    cap = cv2.VideoCapture(input_video_path)

    # 獲取影片資訊
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 開始處理影片
    body_frame_json = []
    frame_number = 0
    last_wrist_point = None
    last_ear_point = None  # 新增：保存上一幀的耳朵位置
    smoothing_factor = 0.3

    # 開始主循環處理影片
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        # 對每一幀執行姿態檢測
        body_frame_results = yolo_pose_model(frame)
        tennis_ball_results  = tennis_ball_model(frame)

        # 初始化幀身體數據
        body_frame_data = {
            "frame": frame_number,
            "right_wrist": {"x": None, "y": None},
            "right_wrist_vector": {"x": None, "y": None},
            "right_shoulder": {"x": None, "y": None},
            "right_hip": {"x": None, "y": None},
            "right_ear": {"x": None, "y": None},
            "right_ear_vector": {"x": None, "y": None},  # 新增：right_ear_vector
            "tennis_ball": {"x": None, "y": None},
            "tennis_ball_hit": False
        }

        # 處理影片檢測結果
        for result in body_frame_results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()
                if keypoints.shape[0] > 10:
                    right_wrist = tuple(map(int, keypoints[10][:2]))
                    right_shoulder = tuple(map(int, keypoints[6][:2]))
                    right_hip = tuple(map(int, keypoints[12][:2]))
                    right_ear = tuple(map(int, keypoints[4][:2]))

                    # 平滑和記錄軌跡點
                    smoothed_wrist = smooth_point(last_wrist_point, right_wrist, smoothing_factor)
                    smoothed_ear = smooth_point(last_ear_point, right_ear, smoothing_factor)  # 新增：平滑耳朵位置
                    
                    body_frame_data["right_wrist"]["x"] = int(smoothed_wrist[0])
                    body_frame_data["right_wrist"]["y"] = int(smoothed_wrist[1])
                    body_frame_data["right_shoulder"]["x"], body_frame_data["right_shoulder"]["y"] = right_shoulder
                    body_frame_data["right_hip"]["x"], body_frame_data["right_hip"]["y"] = right_hip
                    body_frame_data["right_ear"]["x"], body_frame_data["right_ear"]["y"] = int(smoothed_ear[0]), int(smoothed_ear[1])

                    # 計算揮拍軌跡運動向量差，以及平滑揮拍軌跡
                    if last_wrist_point:
                        wrist_vector = (smoothed_wrist[0] - last_wrist_point[0], smoothed_wrist[1] - last_wrist_point[1])
                        body_frame_data["right_wrist_vector"]["x"], body_frame_data["right_wrist_vector"]["y"] = wrist_vector

                    # 計算揮拍軌跡運動向量差
                    if last_ear_point:
                        ear_vector = (smoothed_ear[0] - last_ear_point[0], smoothed_ear[1] - last_ear_point[1])
                        body_frame_data["right_ear_vector"]["x"], body_frame_data["right_ear_vector"]["y"] = ear_vector

                    last_wrist_point = smoothed_wrist
                    last_ear_point = smoothed_ear  # 新增：更新上一幀的耳朵位置

        # 處理網球檢測結果（保持不變）
        for result in tennis_ball_results:
            for det in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, det[:4])
                cls = int(result.boxes.cls[0])
                conf = float(result.boxes.conf[0])
                label = result.names[cls]
                
                if label in ["ball", "sports ball"] and conf > 0.5:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    body_frame_data["tennis_ball"]["x"] = center_x
                    body_frame_data["tennis_ball"]["y"] = center_y
                    break

        body_frame_json.append(body_frame_data)
        frame_number += 1

        # 記錄數據並更新進度
        if frame_number % 10 == 0:
            print(f"Processing frame {frame_number}")
    
    cap.release()

    # -------------step2: 判斷開始揮拍和節數揮拍的偵數-------------

    swing_start, swing_end, ball_return = determine_swing_trajectory(body_frame_json)

    # 把開始揮拍到節數揮拍的向量數據寫進json裡面
    filtered_body_frame_json = [body_frame_data for body_frame_data in body_frame_json if swing_start <= body_frame_data['frame'] <= swing_end]
    for frame_data in filtered_body_frame_json:
        if frame_data['frame'] == ball_return:
            frame_data['tennis_ball_hit'] = True
            break
    save_json_to_jsonfile(filtered_body_frame_json, output_json_path)
    
    # -------------step3: 把判斷的軌跡畫上去影片，並輸出-------------

    # 重新打開視頻以繪製軌跡
    cap = cv2.VideoCapture(input_video_path)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 初始化幀計數器和軌跡點列表
    wrist_trajectory_points = []
    ball_trajectory_points = []
    wrist_trail_thickness = 5  # 手腕軌跡厚度
    ball_trail_thickness = 3   # 網球軌跡厚度
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 檢查當前幀是否在我們感興趣的範圍內
        if swing_start <= frame_number <= swing_end:
            # 從過濾後的 JSON 數據中獲取當前幀的點坐標
            current_frame_data = next((item for item in filtered_body_frame_json if item["frame"] == frame_number), None)
            if current_frame_data:
                wrist_point = (current_frame_data['right_wrist']['x'], current_frame_data['right_wrist']['y'])
                ball_point = (current_frame_data['tennis_ball']['x'], current_frame_data['tennis_ball']['y'])
                
                # 如果坐標有效，將其添加到相應的軌跡點列表中
                if wrist_point[0] is not None and wrist_point[1] is not None:
                    wrist_trajectory_points.append(wrist_point)
                if ball_point[0] is not None and ball_point[1] is not None:
                    ball_trajectory_points.append(ball_point)

        # 如果有多於一個點，在幀上繪製手腕軌跡線
        if len(wrist_trajectory_points) > 1:
            cv2.polylines(frame, [np.array(wrist_trajectory_points)], False, (0, 255, 255), wrist_trail_thickness)

        # 如果有多於一個點，在幀上繪製網球軌跡線
        if len(ball_trajectory_points) > 1:
            cv2.polylines(frame, [np.array(ball_trajectory_points)], False, (255, 0, 0), ball_trail_thickness)

        # 在當前點上繪製圓圈，標記最新的位置
        if frame_number <= swing_end:
            if wrist_trajectory_points:
                cv2.circle(frame, wrist_trajectory_points[-1], wrist_trail_thickness + 2, (0, 0, 255), -1)
            if ball_trajectory_points:
                cv2.circle(frame, ball_trajectory_points[-1], ball_trail_thickness + 2, (255, 0, 0), -1)

        # 添加秒數和幀數信息到影片左上角
        seconds = frame_number / fps
        cv2.putText(frame, f"Time: {seconds:.2f}s Frame: {frame_number}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 將處理後的幀寫入輸出影片
        out.write(frame)

        frame_number += 1

        # 每處理 10 幀就顯示一次進度
        if frame_number % 10 == 0:
            print(f"Drawing trajectory: Processing frame {frame_number}")

    # 釋放資源
    cap.release()
    out.release()





def compare_trajectories(input_video_path, output_video_path, output_video_path2, output_json_path, output_json_path2, answer_file):
    
    # -------------step1: 將對比的軌跡座標等比縮放-------------

    answer_json_data = get_json_from_jsonfile(answer_file)
    output_json_data = get_json_from_jsonfile(output_json_path)

    # 原本數據的基準點
    for json_data in output_json_data:
        tester_shoulder = (json_data['right_shoulder']['x'],json_data['right_shoulder']['y'])
        tester_hip = (json_data['right_hip']['x'],json_data['right_hip']['y'])
        if 1 == 1:
            break

    # 答案數據的基準點
    for json_data in answer_json_data:
        answer_shoulder = (json_data['right_shoulder']['x'],json_data['right_shoulder']['y'])
        answer_hip = (json_data['right_hip']['x'],json_data['right_hip']['y'])
        if 1 == 1:
            break

    tester_distance = calculate_distance(tester_shoulder, tester_hip)
    answer_distance = calculate_distance(answer_shoulder, answer_hip)
    scale_ratio = tester_distance / answer_distance
    answer_scaled_json = scale_coordinates(answer_json_data, scale_ratio) # 將對比的json等比縮放
       
    # -------------step2: 將對比的軌跡座標整體移動到影片正確位置-------------

    for json_data in output_json_data:
        bx = int(json_data["right_wrist"]['x'])
        by = int(json_data["right_wrist"]['y'])
        if 1 == 1:
            break

    for json_data in answer_scaled_json:
        ax = int(json_data["right_wrist"]['x'])
        ay = int(json_data["right_wrist"]['y'])
        if 1 == 1:
            break

    distance_x = ax - bx
    distance_y = ay - by

    # 改變座標位置
    for json_data in answer_scaled_json:
        json_data["right_wrist"]['x'] = int(json_data["right_wrist"]['x'])-distance_x
        json_data["right_wrist"]['y'] = int(json_data["right_wrist"]['y'])-distance_y

    
       
    # -------------step3: 將對比的軌跡畫上去影片-------------

    cap = cv2.VideoCapture(output_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path2, fourcc, fps, (frame_width, frame_height))

    trail_color = (0, 0, 255)  # 黃色
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
        if frame_number < len(answer_scaled_json):
            current_data = answer_scaled_json[frame_number]
            current_x = current_data["right_wrist"]['x']
            current_y = current_data["right_wrist"]['y']
            
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
    print(f"總處理幀數：{frame_number}")
    save_json_to_jsonfile(answer_scaled_json, output_json_path2)


def main():
    yolo_pose_model = YOLO('yolov8n-pose.pt') # keypoint
    tennis_ball_model = YOLO('tennis_ball.pt') # objections
    input_video_path = 'test2/Produce_2.mp4'
    # input_video_path = 'answer.mp4'
    output_video_path = f'{input_video_path.replace(".mp4", "")}_trajectory.mp4'
    output_video_path2 = f'{input_video_path.replace(".mp4", "")}_trajectory_comparison.mp4'
    output_json_path = f'{input_video_path.replace(".mp4", "")}_trajectory.json'
    answer_file = 'test/answer.json'
    output_json_path2 = f'{answer_file.replace(".json", "")}_trajectory.json'
    process_video(yolo_pose_model, tennis_ball_model, input_video_path, output_video_path, output_json_path)
    # compare_trajectories(input_video_path, output_video_path, output_video_path2, output_json_path, output_json_path2, answer_file)

if __name__ == "__main__":
    main()