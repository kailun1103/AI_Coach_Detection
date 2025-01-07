import numpy as np
import cv2
import json
import time
from ultralytics import YOLO

def process_video(pose_model, ball_model, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_json = []
    frame_number = 0

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
                    frame_data["left_wrist"].update({"x": left_wrist[0], "y": left_wrist[1]})

        # Get tennis ball coordinates
        for result in ball_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if float(box.conf[0]) > 0.5:
                    frame_data["tennis_ball"].update({
                        "x": (x1 + x2) // 2,
                        "y": (y1 + y2) // 2
                    })
                    break

        frame_json.append(frame_data)
        frame_number += 1

    cap.release()

    # Handle last frame
    if frame_json and frame_json[-1]["left_wrist"]["x"] is None and len(frame_json) > 1:
        frame_json[-1]["left_wrist"] = frame_json[-2]["left_wrist"]

    return frame_json

def analyze_trajectory(pose_model, ball_model, video_path):
    trajectory = process_video(pose_model, ball_model, video_path)
    output_path = video_path.replace('.mp4', '_trajectory.json')
    
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)
    return output_path

if __name__ == "__main__":
    total_start_time = time.time()
    
    # Time model loading
    model_load_start = time.time()
    pose_model = YOLO('model/yolov8n-pose.pt')
    ball_model = YOLO('model/yolov8_side_backhand_v1.pt')
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.8f}s")

    video_path = 'leftBackhand_45.mp4'
    # video_path = 'leftBackhand_side.mp4'

    # Time trajectory analysis
    analysis_start = time.time()
    output_path = analyze_trajectory(pose_model, ball_model, video_path)
    analysis_time = time.time() - analysis_start
    print(f"Trajectory analysis time: {analysis_time:.8f}s")

    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f}s")