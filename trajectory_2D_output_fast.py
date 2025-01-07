import numpy as np
from ultralytics import YOLO
import math
import cv2
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_video_batch_async(yolo_pose_model, ball_model, video_path, batch_size=6):
    """使用異步批量處理的方式處理單個影片"""
    # print(f"Start processing {video_path}...")
    cap = cv2.VideoCapture(video_path)

    # 獲取影片資訊
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 開始處理影片
    frame_json = []
    frame_number = 0
    frames_batch = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            if frames_batch:  # 處理最後剩餘的幀
                await process_batch_async(frames_batch, frame_number - len(frames_batch),
                                          yolo_pose_model, ball_model, frame_json)
            # print(f"Finished processing {video_path}")
            break

        frames_batch.append(frame)

        # 當收集到足夠的幀時進行批量處理
        if len(frames_batch) == batch_size:
            await process_batch_async(frames_batch, frame_number - batch_size + 1,
                                      yolo_pose_model, ball_model, frame_json)
            frames_batch = []  # 清空批次

            # if frame_number % 10 == 0:
            #     print(f"Processing {video_path}: frame {frame_number}")

        frame_number += 1

    cap.release()

    # 處理最後一幀的空值
    if frame_json and frame_json[-1]["left_wrist"]["x"] is None and len(frame_json) > 1:
        frame_json[-1]["left_wrist"] = frame_json[-2]["left_wrist"]

    return frame_json

async def process_batch_async(frames, start_frame, yolo_pose_model, ball_model, frame_json):
    """處理一批幀（異步）"""
    loop = asyncio.get_event_loop()

    # 使用執行緒池來執行同步的模型推理操作
    pose_future = loop.run_in_executor(None, lambda: list(yolo_pose_model(frames, stream=True)))
    ball_future = loop.run_in_executor(None, lambda: list(ball_model(frames, stream=True)))

    pose_results, ball_results = await asyncio.gather(pose_future, ball_future)

    def process_single_frame(i, pose_result, ball_result):
        frame_data = {
            "frame": start_frame + i,
            "left_wrist": {"x": None, "y": None},
            "tennis_ball": {"x": None, "y": None}
        }

        # 處理左手腕座標
        if pose_result.keypoints is not None:
            keypoints = pose_result.keypoints.xy[0].cpu().numpy()
            if keypoints.shape[0] > 10:
                left_wrist = tuple(map(int, keypoints[10][:2]))
                frame_data["left_wrist"]["x"] = int(left_wrist[0])
                frame_data["left_wrist"]["y"] = int(left_wrist[1])

        # 處理網球座標
        for box in ball_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            conf = float(box.conf[0])

            if conf > 0.5:
                frame_data["tennis_ball"]["x"] = center_x
                frame_data["tennis_ball"]["y"] = center_y
                break

        return frame_data

    # 使用執行緒池處理每一幀
    with ThreadPoolExecutor() as executor:
        processed_frames = await loop.run_in_executor(
            executor,
            lambda: [process_single_frame(i, pose_result, ball_result) 
                     for i, (pose_result, ball_result) in enumerate(zip(pose_results, ball_results))]
        )

    frame_json.extend(processed_frames)

async def analyze_trajectory_async(yolo_pose_model, tennis_ball_model, video_path):
    """分析單一影片的軌跡"""
    # 處理影片
    # print(f"Processing video: {video_path}")
    trajectory = await process_video_batch_async(yolo_pose_model, tennis_ball_model, video_path)

    # 儲存軌跡數據
    output_path = video_path.replace('.mp4', '_trajectory.json')
    # print(f"\nSaving trajectory data to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)

    return output_path

if __name__ == "__main__":
    print("Loading models...")
    

    # 載入模型
    yolo_pose_model = YOLO('model/yolov8n-pose.pt')
    tennis_ball_model = YOLO('model/yolov8_side_backhand_v1.pt')

    # 設定影片路徑
    video_path = 'leftBackhand_45.mp4'
    # video_path = 'leftBackhand_side.mp4'
    start_time = time.time()  # 開始計時

    # 使用 asyncio 執行異步軌跡計算
    asyncio.run(analyze_trajectory_async(yolo_pose_model, tennis_ball_model, video_path))

    # 計算並顯示執行時間
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")