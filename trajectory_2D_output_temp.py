import cv2
import json
import time
import torch
import numpy as np
from ultralytics import YOLO

def process_video(pose_model, ball_model, video_path, batch_size=16, conf_thres=0.2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return []

    frame_json = []
    frame_number = 0
    
    # Define keypoint names according to YOLOv8-pose output
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # 暫存一批影像
    frames_batch = []
    frame_indices = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_batch.append(frame)
        frame_indices.append(frame_number)
        frame_number += 1

        # 若達到批次大小，或影片結束，就做一次批次推理
        if len(frames_batch) == batch_size:
            frame_json.extend(
                run_inference_on_batch(
                    pose_model, 
                    ball_model, 
                    frames_batch, 
                    frame_indices, 
                    keypoint_names, 
                    conf_thres
                )
            )
            frames_batch.clear()
            frame_indices.clear()

    # 如果最後剩下一些影像未推理，做最後一次推理
    if frames_batch:
        frame_json.extend(
            run_inference_on_batch(
                pose_model,
                ball_model,
                frames_batch,
                frame_indices,
                keypoint_names,
                conf_thres
            )
        )

    cap.release()

    # 如果需要處理「最後一幀關鍵點缺漏」的情況，可在這裡進行
    # 例如：複製前一幀的關鍵點
    if len(frame_json) > 1:
        last_frame = frame_json[-1]
        prev_frame = frame_json[-2]
        for keypoint in keypoint_names:
            if last_frame[keypoint]["x"] is None:
                last_frame[keypoint] = prev_frame[keypoint]

    return frame_json


def run_inference_on_batch(
    pose_model,
    ball_model,
    frames_batch,
    frame_indices,
    keypoint_names,
    conf_thres
):
    """
    對一批影像進行一次推理，並回傳結果 JSON list
    """
    # ------ Pose 推理 ------
    # 可在這裡設定 device、半精度等參數，例如：pose_model.predict(frames_batch, device=0, half=True)
    pose_results = pose_model.predict(frames_batch, verbose=False, conf=0.25)
    
    # ------ Ball 推理 ------
    ball_results = ball_model.predict(frames_batch, verbose=False, conf=0.25)

    batch_output = []

    # 分別對應各張影像的結果
    for i, (pose_res, ball_res, idx) in enumerate(zip(pose_results, ball_results, frame_indices)):
        frame_data = {"frame": idx, "tennis_ball": {"x": None, "y": None}}
        
        # 預設所有關鍵點為 None
        for kp in keypoint_names:
            frame_data[kp] = {"x": None, "y": None}

        # 取得姿勢關鍵點
        if pose_res.keypoints is not None:
            # YOLOv8 Pose 一張圖可能偵測到多個人，假設我們只取第一個人 (index=0)
            # 若要處理多個人，需要進一步處理
            kpts_xy = pose_res.keypoints.xy
            if len(kpts_xy) > 0:
                # 只取第一個人的關鍵點
                keypoints = kpts_xy[0].cpu().numpy()
                # 與 keypoint_names 順序對應
                for kp_idx, kp_name in enumerate(keypoint_names):
                    x, y = keypoints[kp_idx][:2]
                    # 如果此點偵測結果並非 (0,0)，則紀錄
                    frame_data[kp_name] = {
                        "x": int(x) if x > 0 else None,
                        "y": int(y) if y > 0 else None,
                    }

        # 取得網球位置
        # 同樣可能有多個框，這裡只取信心度最高或第一個超過閾值的框
        boxes = ball_res.boxes
        if boxes is not None and len(boxes) > 0:
            # 取信心度最高的一個或只要大於 conf_thres
            for box in boxes:
                if float(box.conf[0]) >= conf_thres:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    frame_data["tennis_ball"] = {
                        "x": (x1 + x2) // 2,
                        "y": (y1 + y2) // 2,
                    }
                    # 如果只取第一顆球，break
                    break

        batch_output.append(frame_data)

    return batch_output


def analyze_trajectory(pose_model, ball_model, video_path, batch_size=16):
    # 主要流程：以批次處理影片
    trajectory = process_video(pose_model, ball_model, video_path, batch_size=batch_size)

    output_path = video_path.replace('.mp4', '(2D_trajectory).json')
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)

    return output_path


if __name__ == "__main__":
    total_start_time = time.time()

    # 檢查 GPU 是否可用
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # 啟動計時器：Model load
    model_load_start = time.time()
    pose_model = YOLO('model/yolov8n-pose.pt')
    ball_model = YOLO('model/tennisball_OD_v1.pt')

    # 也可以直接指定預設 device
    pose_model.to(device)
    ball_model.to(device)

    # 若想測試半精度可嘗試（需視模型與硬體是否支援）
    # pose_model.predict(..., half=True)
    # ball_model.predict(..., half=True)

    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.8f}s")

    video_path = 'pro_1_1_45.mp4'

    # Trajectory analysis
    analysis_start = time.time()

    # 以批次大小 16 來處理，可依 GPU 能力調整
    output_path = analyze_trajectory(pose_model, ball_model, video_path, batch_size=8)

    analysis_time = time.time() - analysis_start
    print(f"Trajectory analysis time: {analysis_time:.8f}s")

    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f}s")