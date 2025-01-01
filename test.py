import numpy as np
import cv2
import json
import time
from ultralytics import YOLO
from scipy.interpolate import interp1d

class CustomEncoder(json.JSONEncoder):
    """自定義 JSON 編碼器來保持數字格式"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def interpolate_coordinates(trajectory_data, key):
    """使用插值來填補缺失的座標，並保持原始格式"""
    frames = []
    x_coords = []
    y_coords = []
    
    # 收集所有有效的座標點
    for frame in trajectory_data:
        if frame[key]["x"] is not None and frame[key]["y"] is not None:
            frames.append(frame["frame"])
            x_coords.append(frame[key]["x"])
            y_coords.append(frame[key]["y"])
    
    if len(frames) < 2:  # 需要至少兩個點才能進行插值
        return trajectory_data
    
    # 創建插值函數
    x_interp = interp1d(frames, x_coords, kind='linear', fill_value='extrapolate')
    y_interp = interp1d(frames, y_coords, kind='linear', fill_value='extrapolate')
    
    # 填充插值結果
    interpolated_data = []
    for frame in trajectory_data:
        frame_number = frame["frame"]
        new_frame = {
            "frame": frame_number,
            "left_wrist": {
                "x": frame["left_wrist"]["x"],
                "y": frame["left_wrist"]["y"]
            },
            "tennis_ball": {
                "x": frame["tennis_ball"]["x"],
                "y": frame["tennis_ball"]["y"]
            }
        }
        
        # 如果當前key的座標是缺失的，使用插值結果
        if frame[key]["x"] is None or frame[key]["y"] is None:
            # 確保只在有效範圍內進行插值
            if min(frames) <= frame_number <= max(frames):
                try:
                    x_val = x_interp(frame_number)
                    y_val = y_interp(frame_number)
                    new_frame[key]["x"] = int(x_val)
                    new_frame[key]["y"] = int(y_val)
                except (ValueError, TypeError) as e:
                    continue
        
        interpolated_data.append(new_frame)
    
    return interpolated_data

def process_video(pose_model, ball_model, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_json = []
    frame_number = 0
    
    # 追蹤變數
    first_ball_detection = None
    last_ball_detection = None
    first_wrist_detection = None
    last_wrist_detection = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        body_results = pose_model(frame, verbose=False)
        ball_results = ball_model(frame, verbose=False)

        frame_data = {
            "frame": frame_number,
            "left_wrist": {"x": None, "y": None},
            "tennis_ball": {"x": None, "y": None}
        }

        # Get left wrist coordinates
        for result in body_results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()
                if keypoints.shape[0] > 10:
                    left_wrist = tuple(map(int, keypoints[10][:2]))
                    frame_data["left_wrist"]["x"] = left_wrist[0]
                    frame_data["left_wrist"]["y"] = left_wrist[1]
                    if first_wrist_detection is None:
                        first_wrist_detection = frame_number
                    last_wrist_detection = frame_number

        # Get tennis ball coordinates
        for result in ball_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if float(box.conf[0]) > 0.5:
                    frame_data["tennis_ball"]["x"] = (x1 + x2) // 2
                    frame_data["tennis_ball"]["y"] = (y1 + y2) // 2
                    if first_ball_detection is None:
                        first_ball_detection = frame_number
                    last_ball_detection = frame_number
                    break

        frame_json.append(frame_data)
        frame_number += 1

    cap.release()
    
    # 對缺失的座標進行插值
    try:
        interpolated_json = interpolate_coordinates(frame_json, "tennis_ball")
        interpolated_json = interpolate_coordinates(interpolated_json, "left_wrist")
    except Exception as e:
        print(f"插值處理時發生錯誤: {e}")
        return frame_json, {
            "ball": {
                "first_detection": first_ball_detection,
                "last_detection": last_ball_detection
            },
            "wrist": {
                "first_detection": first_wrist_detection,
                "last_detection": last_wrist_detection
            }
        }

    return interpolated_json, {
        "ball": {
            "first_detection": first_ball_detection,
            "last_detection": last_ball_detection
        },
        "wrist": {
            "first_detection": first_wrist_detection,
            "last_detection": last_wrist_detection
        }
    }

def analyze_trajectory(pose_model, ball_model, video_path):
    trajectory, detection_info = process_video(pose_model, ball_model, video_path)
    output_path = video_path.replace('.mp4', '_trajectory_interpolated.json')
    
    # 計算影片FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # 轉換幀數為時間
    ball_first_time = detection_info["ball"]["first_detection"] / fps if detection_info["ball"]["first_detection"] is not None else None
    ball_last_time = detection_info["ball"]["last_detection"] / fps if detection_info["ball"]["last_detection"] is not None else None
    wrist_first_time = detection_info["wrist"]["first_detection"] / fps if detection_info["wrist"]["first_detection"] is not None else None
    wrist_last_time = detection_info["wrist"]["last_detection"] / fps if detection_info["wrist"]["last_detection"] is not None else None
    
    # 直接使用 trajectory 作為輸出，不包裝在額外的 key 中
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2, cls=CustomEncoder)
    
    return output_path, detection_info, ball_first_time, ball_last_time, wrist_first_time, wrist_last_time

if __name__ == "__main__":
    total_start_time = time.time()
    
    model_load_start = time.time()
    pose_model = YOLO('model/yolov8n-pose.pt')
    ball_model = YOLO('model/yolov8_side_backhand_v1.pt')
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.8f}s")

    video_path = 'leftBackhand_side.mp4'

    analysis_start = time.time()
    output_path, detection_info, ball_first_time, ball_last_time, wrist_first_time, wrist_last_time = analyze_trajectory(pose_model, ball_model, video_path)
    analysis_time = time.time() - analysis_start
    
    print("\n分析結果（包含插值）:")
    print("網球偵測:")
    print(f"  首次偵測到網球的幀數: {detection_info['ball']['first_detection']}")
    print(f"  最後偵測到網球的幀數: {detection_info['ball']['last_detection']}")
    print(f"  首次偵測時間: {ball_first_time:.2f}s" if ball_first_time is not None else "  無偵測資料")
    print(f"  最後偵測時間: {ball_last_time:.2f}s" if ball_last_time is not None else "  無偵測資料")
    
    print("\n左手腕偵測:")
    print(f"  首次偵測到左手腕的幀數: {detection_info['wrist']['first_detection']}")
    print(f"  最後偵測到左手腕的幀數: {detection_info['wrist']['last_detection']}")
    print(f"  首次偵測時間: {wrist_first_time:.2f}s" if wrist_first_time is not None else "  無偵測資料")
    print(f"  最後偵測時間: {wrist_last_time:.2f}s" if wrist_last_time is not None else "  無偵測資料")
    
    print(f"\nTrajectory analysis time: {analysis_time:.8f}s")
    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f}s")