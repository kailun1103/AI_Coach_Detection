import numpy as np
import cv2
import json
import time
from ultralytics import YOLO

def process_video(ball_model, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_json = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        ball_results = ball_model(frame, verbose=False)

        frame_data = {
            "frame": frame_number,
            "tennis_ball": {"x": None, "y": None}
        }

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
    return frame_json

def analyze_trajectory(ball_model, video_path):
    trajectory = process_video(ball_model, video_path)
    output_path = video_path.replace('.mp4', '_trajectory.json')
    
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)
    return output_path

if __name__ == "__main__":
    start_time = time.time()
    
    ball_model = YOLO('basketball.pt')
    video_path = 'left.mp4'

    output_path = analyze_trajectory(ball_model, video_path)

    print(f"Execution time: {time.time() - start_time:.2f}s")