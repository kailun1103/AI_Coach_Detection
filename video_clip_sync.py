import cv2
import json
import numpy as np
import time
from tqdm import tqdm

def get_hit_frame(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    for frame_info in data:
        if frame_info.get("tennis_ball_hit", False):
            return frame_info["frame"]
    return None

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps
    cap.release()
    return total_frames, fps, duration

def synchronize_videos(input_path_1, input_path_2, output_path_1, output_path_2, json_path_1, json_path_2, trim_length=60):
    frames1, fps1, duration1 = get_video_info(input_path_1)
    frames2, fps2, duration2 = get_video_info(input_path_2)
    
    print("\n原始影片資訊:")
    print(f"影片 1: {frames1} 幀, {duration1:.2f} 秒")
    print(f"影片 2: {frames2} 幀, {duration2:.2f} 秒")

    hit_frame_1 = get_hit_frame(json_path_1)
    hit_frame_2 = get_hit_frame(json_path_2)
    
    print(f"\n擊球幀位置:")
    print(f"影片 1: 第 {hit_frame_1} 幀")
    print(f"影片 2: 第 {hit_frame_2} 幀")

    max_frames_after = min(frames1 - hit_frame_1, frames2 - hit_frame_2)
    max_frames_before = min(hit_frame_1, hit_frame_2)
    
    frames_before = min(trim_length // 2, max_frames_before)
    frames_after = min(trim_length - frames_before, max_frames_after)
    
    start_frame_1 = hit_frame_1 - frames_before
    start_frame_2 = hit_frame_2 - frames_before
    
    end_frame_1 = hit_frame_1 + frames_after
    end_frame_2 = hit_frame_2 + frames_after

    print(f"\n剪輯資訊:")
    print(f"影片 1: 從第 {start_frame_1} 幀到第 {end_frame_1} 幀")
    print(f"影片 2: 從第 {start_frame_2} 幀到第 {end_frame_2} 幀")
    
    cap1 = cv2.VideoCapture(input_path_1)
    cap2 = cv2.VideoCapture(input_path_2)
    
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(output_path_1, fourcc, fps1, (width1, height1))
    out2 = cv2.VideoWriter(output_path_2, fourcc, fps2, (width2, height2))
    
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame_1)
    print("\n處理影片 1...")
    frames_to_process = end_frame_1 - start_frame_1
    for _ in tqdm(range(frames_to_process)):
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        out1.write(frame1)
    
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame_2)
    print("處理影片 2...")
    for _ in tqdm(range(frames_to_process)):
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        out2.write(frame2)
    
    final_duration = frames_to_process / fps1
    print(f"\n最終影片資訊:")
    print(f"兩個影片都是 {frames_to_process} 幀, {final_duration:.2f} 秒")
    
    cap1.release()
    cap2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()
    print("\n同步完成!")

if __name__ == "__main__":
    start_time = time.time()
    
    input_video_1 = "pro_1_1_45.mp4"
    input_video_2 = "pro_1_1_side.mp4"
    output_video_1 = "pro_1_1_45_完成.mp4"
    output_video_2 = "pro_1_1_side_完成.mp4"
    json_path_1 = "pro_1_1_45(2D_trajectory_smoothed).json"
    json_path_2 = "pro_1_1_side(2D_trajectory_smoothed).json"

    print("開始執行影片同步...")
    synchronize_videos(input_video_1, input_video_2, 
                      output_video_1, output_video_2,
                      json_path_1, json_path_2)
    
    print(f"執行時間: {time.time() - start_time:.4f}秒")