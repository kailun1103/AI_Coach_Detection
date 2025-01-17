import time
import numpy as np 
from ultralytics import YOLO
from trajectory_2D_output import analyze_trajectory
from trajector_2D_smoothing import smooth_2D_trajectory
from trajectory_3D_output import process_trajectories
from trajector_3D_smoothing import smooth_3D_trajectory
from trajector_2D_sync import sync_trajectories
from drawing_3D import create_3d_plots

# Start calculating total execution time
start_total = time.perf_counter()

# Input videos
video_side = 'temp/junior_side.mp4'
video_45 = 'temp/junior_45.mp4'

# junior backhand
# P1 = np.array([ # Left camera (main)
#     [5830.127771, 0, 2707.891358, 0],
#     [0, 5660.852212, 2650.794043, 0],
#     [0, 0, 1, 0] 
# ])

# P2 = np.array([ # 45-degree camera
#     [-127.726676, -549.005678, 4533.763086, -23449322.445458],
#     [-1883.494533, 3034.703903, 1416.289936, 6432610.718249],
#     [-0.860417, -0.091385, 0.501330, 2218.320368]
# ])

# junior forehand
P1 = np.array([ # Left camera (main)
    [4868.506691,    0.000000, 2819.088860,    0.000000],
    [   0.000000, 3887.239287, 2362.952860,    0.000000],
    [   0.000000,    0.000000,    1.000000,    0.000000]
])

P2 = np.array([ # 45-degree camera
    [    -1532.746717,       704.787489,      4054.256764, -19560781.953567],
    [    -2370.947477,      3331.290729,       326.229463,   9897228.332878],
    [       -0.944408,         0.080699,         0.318719,      3568.669508]
])

# pro forehand
# P1 = np.array([ # Left camera (main)
#     [2259.233089,    0.000000, 2765.855088,    0.000000],
#     [   0.000000, 2262.229625, 2527.097657,    0.000000],
#     [   0.000000,    0.000000,    1.000000,    0.000000]
# ])

# P2 = np.array([ # 45-degree camera
#     [     133.791680,     -550.740908,     3565.586369, -2967047.706145],
#     [   -1544.950048,     1821.751422,     2014.789599,  3406954.592979],
#     [      -0.593895,       -0.198783,        0.779598,     1344.472848]
# ])

# Load models
print("Step 1: Loading models...")
start_model = time.perf_counter()
yolo_pose_model = YOLO('model/yolov8n-pose.pt')
yolo_tennis_ball_model = YOLO('model/yolov8_side_backhand_v1.pt')
model_time = time.perf_counter() - start_model
print(f"-- Model loading completed, time taken: {model_time:.4f} seconds")

# 2D trajectory analysis
print("\nStep 2: Analyzing 2D trajectories...")
start_2d = time.perf_counter()
trajectory_side = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_side)
trajectory_45 = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_45)
trajectory_2d_time = time.perf_counter() - start_2d
print(f"-- 2D trajectory analysis completed, time taken: {trajectory_2d_time:.4f} seconds")

# 2D trajectory smoothing/interpolation/hitting angle
print("\nStep 3: Smoothing/interpolation/hitting_angle 2D trajectories...")
start_smooth_2d = time.perf_counter()
trajectory_side_smoothing = smooth_2D_trajectory(trajectory_side)
trajectory_45_smoothing = smooth_2D_trajectory(trajectory_45)
smooth_2d_time = time.perf_counter() - start_smooth_2d
print(f"-- 2D smoothing completed, time taken: {smooth_2d_time:.4f} seconds")


# 2D trajectory synchronous
print("\nStep 4: trajectory synchronous...")
start_sync = time.perf_counter()
sync_trajectories(trajectory_side_smoothing, trajectory_45_smoothing)
trajectory_sync_time = time.perf_counter() - start_sync
print(f"-- 2D trajectory synchronous completed, time taken: {trajectory_sync_time:.4f} seconds")


# 3D trajectory analysis
print("\nStep 5: Calculating 3D trajectories...")
start_3d = time.perf_counter()
trajectory_3d = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)
trajectory_3d_time = time.perf_counter() - start_3d
print(f"-- 3D trajectory calculation completed, time taken: {trajectory_3d_time:.4f} seconds")


# 3D trajectory smoothing
print("\nStep 6: Smoothing 3D trajectories...")
start_smooth_3d = time.perf_counter()
trajectory_3d_smoothing = smooth_3D_trajectory(trajectory_3d)
smooth_3d_time = time.perf_counter() - start_smooth_3d
print(f"-- 3D smoothing completed, time taken: {smooth_3d_time:.4f} seconds")


# Execution time summary
total_time = time.perf_counter() - start_total

print('-'*50)
print(f"Total execution time:       {total_time:>10.4f} sec")
print('='*50)