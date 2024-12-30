import time
import numpy as np 
from ultralytics import YOLO
from trajectory_2D_output_nor import analyze_trajectory
from trajector_2D_smoothing import smooth_2D_trajectory
from trajectory_3D_output import process_trajectories
from trajector_3D_smoothing import smooth_3D_trajectory

# input video
video_left = 'leftBackhand_side.mp4'
video_45 = 'leftBackhand_45.mp4'

# input_projection_matrix
P1 = np.array([ # left camera(main)
    [5830.127771, 0, 2707.891358, 0],
    [0, 5660.852212, 2650.794043, 0],
    [0, 0, 1, 0] 
])

P2 = np.array([ # 45 camera
    [-127.726676, -549.005678, 4533.763086, -23449322.445458],
    [-1883.494533, 3034.703903, 1416.289936, 6432610.718249],
    [-0.860417, -0.091385, 0.501330, 2218.320368]
])

# Load models
start_model = time.perf_counter()
yolo_pose_model = YOLO('model/yolov8n-pose.pt')
yolo_tennis_ball_model = YOLO('model/yolov8_side_backhand_v1.pt')
print(f"-- Model loading time: {time.perf_counter() - start_model:.4f}s")
start_load = time.perf_counter()

# Get 2D trajectories
start_2d = time.perf_counter()
trajectory_side = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_left)
trajectory_45 = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_45)

# Smooth 2D trajectories 
start_smooth_2d = time.perf_counter()
trajectory_side_smoothing = smooth_2D_trajectory(trajectory_side)
trajectory_45_smoothing = smooth_2D_trajectory(trajectory_45)

# Calculate 3D trajectory
start_3d = time.perf_counter()
trajectory_3d = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)

# Smooth 3D trajectory
start_smooth_3d = time.perf_counter()
trajectory_3d_smoothing = smooth_3D_trajectory(trajectory_3d)

total_time = time.perf_counter() - start_load
print('-'*40)
print(f"-- 2D trajectory calculation time: {time.perf_counter() - start_2d:.4f}s")
print(f"-- 2D smoothing time: {time.perf_counter() - start_smooth_2d:.4f}s")
print(f"-- 3D trajectory calculation time: {time.perf_counter() - start_3d:.4f}s")
print(f"-- 3D smoothing time: {time.perf_counter() - start_smooth_3d:.4f}s")
print(f"Total execution time: {total_time:.4f}s")