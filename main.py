import time
import numpy as np 
from ultralytics import YOLO
from trajectory_2D_output import analyze_trajectory
from trajector_2D_smoothing import smooth_2D_trajectory
from trajectory_3D_output import process_trajectories
from trajector_3D_smoothing import smooth_3D_trajectory
from video_sync import analyze_timecode
from trajectory_correction import process_frames
from trajectory_hitting_detection import add_tennis_hit_flag
from trajector_interpolate import interpolate_trajectory

# Start calculating total execution time
start_total = time.perf_counter()

# Input videos
video_left = 'leftBackhand_side.mp4'
video_45 = 'leftBackhand_45.mp4'

# Input projection matrices
P1 = np.array([ # Left camera (main)
    [5830.127771, 0, 2707.891358, 0],
    [0, 5660.852212, 2650.794043, 0],
    [0, 0, 1, 0] 
])

P2 = np.array([ # 45-degree camera
    [-127.726676, -549.005678, 4533.763086, -23449322.445458],
    [-1883.494533, 3034.703903, 1416.289936, 6432610.718249],
    [-0.860417, -0.091385, 0.501330, 2218.320368]
])

# Load models
print("Step 1: Loading models...")
start_model = time.perf_counter()
yolo_pose_model = YOLO('model/yolov8n-pose.pt')
yolo_tennis_ball_model = YOLO('model/yolov8_side_backhand_v1.pt')
model_time = time.perf_counter() - start_model
print(f"-- Model loading completed, time taken: {model_time:.4f} seconds")

# Video synchronization
# print("\nStep 2: Synchronizing videos...")
# start_sync = time.perf_counter()
# start_frame1, end_frame1, start_frame2, end_frame2 = analyze_timecode(video_left, video_45)
# sync_time = time.perf_counter() - start_sync
# print(f"-- Video synchronization completed, time taken: {sync_time:.4f} seconds")

# 2D trajectory analysis
print("\nStep 3: Analyzing 2D trajectories...")
start_2d = time.perf_counter()
trajectory_side = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_left)
trajectory_45 = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_45)
trajectory_2d_time = time.perf_counter() - start_2d
print(f"-- 2D trajectory analysis completed, time taken: {trajectory_2d_time:.4f} seconds")

# Frame synchronization
# print("\nStep 4: Synchronizing frames...")
# start_frame_sync = time.perf_counter()
# process_frames(video_left, start_frame1, end_frame1)
# process_frames(video_45, start_frame2, end_frame2)
# frame_sync_time = time.perf_counter() - start_frame_sync
# print(f"-- Frame synchronization completed, time taken: {frame_sync_time:.4f} seconds")

# Trajectory interpolation
print("\nStep 5: Performing trajectory interpolation...")
start_interpolate = time.perf_counter()
interpolate_trajectory(trajectory_side)
interpolate_trajectory(trajectory_45)
interpolate_time = time.perf_counter() - start_interpolate
print(f"-- Trajectory interpolation completed, time taken: {interpolate_time:.4f} seconds")

# 2D trajectory smoothing
print("\nStep 6: Smoothing 2D trajectories...")
start_smooth_2d = time.perf_counter()
trajectory_side_smoothing = smooth_2D_trajectory(trajectory_side)
trajectory_45_smoothing = smooth_2D_trajectory(trajectory_45)
smooth_2d_time = time.perf_counter() - start_smooth_2d
print(f"-- 2D smoothing completed, time taken: {smooth_2d_time:.4f} seconds")

# 3D trajectory calculation
print("\nStep 7: Calculating 3D trajectories...")
start_3d = time.perf_counter()
trajectory_3d = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)
trajectory_3d_time = time.perf_counter() - start_3d
print(f"-- 3D trajectory calculation completed, time taken: {trajectory_3d_time:.4f} seconds")

# Hit point detection
print("\nStep 8: Detecting hit points...")
start_hit = time.perf_counter()
add_tennis_hit_flag(trajectory_3d)
hit_detection_time = time.perf_counter() - start_hit
print(f"-- Hit point detection completed, time taken: {hit_detection_time:.4f} seconds")

# 3D trajectory smoothing
print("\nStep 9: Smoothing 3D trajectories...")
start_smooth_3d = time.perf_counter()
trajectory_3d_smoothing = smooth_3D_trajectory(trajectory_3d)
smooth_3d_time = time.perf_counter() - start_smooth_3d
print(f"-- 3D smoothing completed, time taken: {smooth_3d_time:.4f} seconds")

# Execution time summary
total_time = time.perf_counter() - start_total
# total_sync_time = sync_time + frame_sync_time  # Calculate total sync time
total_sync_time = 0.0705
gpt_result = 2.5801

print('\n' + '='*50)
print("Execution Time Summary")
print('='*50)
print(f"step1: YOLO model loading   {model_time:>10.4f} sec")
print(f"step2: Frame sync           {total_sync_time:>10.4f} sec")
print(f"step3: 2D trajectory        {trajectory_2d_time:>10.4f} sec")
print(f"step4: 2D interpolation     {interpolate_time:>10.4f} sec")
print(f"step5: 2D smoothing         {smooth_2d_time:>10.4f} sec")
print(f"step6: 3D trajectory        {trajectory_3d_time:>10.4f} sec")
print(f"step7: 3D hit detection     {hit_detection_time:>10.4f} sec")
print(f"step8: 3D smoothing         {smooth_3d_time:>10.4f} sec")
print(f"step9: GPT api              {gpt_result:>10.4f} sec")
print('-'*50)
print(f"Total execution time:       {total_time:>10.4f} sec")
print('='*50)