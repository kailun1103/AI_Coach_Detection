import time
import numpy as np 
from ultralytics import YOLO
from trajectory_2D_output import analyze_trajectory
from trajector_2D_smoothing import smooth_2D_trajectory
from trajectory_3D_output import process_trajectories
from trajector_3D_smoothing import smooth_3D_trajectory
from trajector_2D_sync import sync_trajectories
from drawing_3D_plotly import create_3d_plots
from drawing_2D_plotly import create_2d_plots

# Start calculating total execution time
start_total = time.perf_counter()

# Load models
print("Step 1: Loading models...")
start_model = time.perf_counter()
yolo_pose_model = YOLO('model/yolov8n-pose.pt')
yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
model_time = time.perf_counter() - start_model
print(f"-- Model loading completed, time taken: {model_time:.4f} seconds")


for i in range(1, 17):
    # Input videos
    video_side = f'pro/pro_1/pro_1_{i}/pro_side_1_{i}.mp4'
    video_45 = f'pro/pro_1/pro_1_{i}/pro_45_1_{i}.mp4'

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
    # P1 = np.array([ # Left camera (main)
    #     [4868.506691,    0.000000, 2819.088860,    0.000000],
    #     [   0.000000, 3887.239287, 2362.952860,    0.000000],
    #     [   0.000000,    0.000000,    1.000000,    0.000000]
    # ])

    # P2 = np.array([ # 45-degree camera
    #     [    -1532.746717,       704.787489,      4054.256764, -19560781.953567],
    #     [    -2370.947477,      3331.290729,       326.229463,   9897228.332878],
    #     [       -0.944408,         0.080699,         0.318719,      3568.669508]
    # ])

    # pro forehand
    P1 = np.array([
        [ 2259.248492,     0.000000,  1651.846528,     0.000000],
        [    0.000000,  2262.230378,  1553.020963,     0.000000],
        [    0.000000,     0.000000,     1.000000,     0.000000],
    ])

    P2 = np.array([
        [  795.771338,  -329.492024,  2697.441025, -4465886.061337],
        [ -966.406397,  2015.459737,  1255.438530, 2097693.969537],
        [   -0.593810,    -0.198914,     0.779630,  1344.552439],
    ])



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

    create_2d_plots(trajectory_side_smoothing)
    create_2d_plots(trajectory_45_smoothing)


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

    create_3d_plots(trajectory_3d_smoothing)


# Execution time summary
total_time = time.perf_counter() - start_total

print('-'*50)
print(f"Total execution time:       {total_time:>10.4f} sec")
print('='*50)