import numpy as np
import cv2
import json
import time
from ultralytics import YOLO

def process_video(pose_model, ball_model, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_json = []
    frame_number = 0
    
    # Define keypoint names according to YOLOv8-pose output
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        body_results = pose_model(frame, verbose=False)
        ball_results = ball_model(frame, verbose=False)

        frame_data = {
            "frame": frame_number,
            "tennis_ball": {"x": None, "y": None}
        }
        
        # Initialize all keypoints as None
        for keypoint in keypoint_names:
            frame_data[keypoint] = {"x": None, "y": None}

        # Get all body keypoints
        for result in body_results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()
                if len(keypoints) == len(keypoint_names):  # Ensure we have all keypoints
                    for idx, keypoint in enumerate(keypoint_names):
                        # Check if the keypoint coordinates are 0.0 (undetected)
                        x, y = keypoints[idx][:2]
                        coords = {
                            "x": int(x) if x != 0.0 else None,
                            "y": int(y) if y != 0.0 else None
                        }
                        frame_data[keypoint].update(coords)

        # Get tennis ball coordinates
        for result in ball_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if float(box.conf[0]) > 0.2: # 信心值
                    frame_data["tennis_ball"].update({
                        "x": (x1 + x2) // 2,
                        "y": (y1 + y2) // 2
                    })
                    break

        frame_json.append(frame_data)
        frame_number += 1

    cap.release()

    # Handle last frame - copy previous frame's keypoints if missing
    if frame_json and len(frame_json) > 1:
        last_frame = frame_json[-1]
        prev_frame = frame_json[-2]
        for keypoint in keypoint_names:
            if last_frame[keypoint]["x"] is None:
                last_frame[keypoint] = prev_frame[keypoint]

    return frame_json

def analyze_trajectory(pose_model, ball_model, video_path):
    trajectory = process_video(pose_model, ball_model, video_path)
    output_path = video_path.replace('.mp4', '(2D_trajectory).json')
    
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)
    return output_path

if __name__ == "__main__":
    total_start_time = time.time()
    
    # Time model loading
    model_load_start = time.time()
    pose_model = YOLO('model/yolov8n-pose.pt')
    ball_model = YOLO('model/tennisball_OD_v1.pt')
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.8f}s")

    video_path = 'pro_1_1_45.mp4'

    # Time trajectory analysis
    analysis_start = time.time()

    output_path = analyze_trajectory(pose_model, ball_model, video_path)

    analysis_time = time.time() - analysis_start
    print(f"Trajectory analysis time: {analysis_time:.8f}s")

    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f}s")