import time
import numpy as np 
from ultralytics import YOLO
from trajectory_2D_output import analyze_trajectory
from trajector_2D_smoothing import smooth_2D_trajectory
from trajectory_3D_output import process_trajectories
from trajector_3D_smoothing import smooth_3D_trajectory
from trajector_2D_sync import sync_trajectories
from drawing_3D_plotly import create_3d_plots
from video_detection import process_video
from video_sync import synchronize_videos
from video_merge import combine_videos_ffmpeg

def process_trajectory(video_side, video_45, yolo_pose_model, yolo_tennis_ball_model):
    # 開始計算總執行時間
    start_total = time.perf_counter()
    timing_results = {}

    # 專業選手正手
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


    # 步驟2：2D軌跡分析
    print("\n步驟2：分析2D軌跡中...")
    start = time.perf_counter()
    trajectory_side = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_side)
    trajectory_45 = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_45)
    timing_results['2D軌跡分析'] = time.perf_counter() - start
    print(f"-- 2D軌跡分析完成，耗時：{timing_results['2D軌跡分析']:.4f} 秒")

    # 步驟3：2D軌跡平滑處理
    print("\n步驟3：進行2D軌跡平滑化/插值/擊球角度處理...")
    start = time.perf_counter()
    trajectory_side_smoothing = smooth_2D_trajectory(trajectory_side)
    trajectory_45_smoothing = smooth_2D_trajectory(trajectory_45)
    timing_results['2D平滑處理'] = time.perf_counter() - start
    print(f"-- 2D平滑處理完成，耗時：{timing_results['2D平滑處理']:.4f} 秒")

    # 步驟4：影片處理
    print("\n步驟4：處理影片中...")
    start = time.perf_counter()
    video_side_processed = process_video(video_side)
    video_45_processed = process_video(video_45)
    timing_results['影片處理'] = time.perf_counter() - start
    print(f"-- 影片處理完成，耗時：{timing_results['影片處理']:.4f} 秒")


    # 步驟5：影片同步
    print("\n步驟5：同步影片中...")
    start = time.perf_counter()
    output_path_1, output_path_2 = synchronize_videos(video_side_processed, video_45_processed, trajectory_side_smoothing, trajectory_45_smoothing)
    timing_results['影片同步'] = time.perf_counter() - start
    print(f"-- 影片同步完成，耗時：{timing_results['影片同步']:.4f} 秒")

    # 步驟6：合併影片
    print("\n步驟6：合併影片中...")
    start = time.perf_counter()
    combine_videos_ffmpeg(output_path_1, output_path_2)
    timing_results['影片合併'] = time.perf_counter() - start
    print(f"-- 影片合併完成，耗時：{timing_results['影片合併']:.4f} 秒")

    # 步驟7：軌跡同步
    print("\n步驟7：同步軌跡中...")
    start = time.perf_counter()
    sync_trajectories(trajectory_side_smoothing, trajectory_45_smoothing)
    timing_results['軌跡同步'] = time.perf_counter() - start
    print(f"-- 軌跡同步完成，耗時：{timing_results['軌跡同步']:.4f} 秒")

    # 步驟8：3D軌跡分析
    print("\n步驟8：計算3D軌跡中...")
    start = time.perf_counter()
    trajectory_3d = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)
    timing_results['3D軌跡分析'] = time.perf_counter() - start
    print(f"-- 3D軌跡計算完成，耗時：{timing_results['3D軌跡分析']:.4f} 秒")

    # 步驟9：3D軌跡平滑處理
    print("\n步驟9：進行3D軌跡平滑處理中...")
    start = time.perf_counter()
    trajectory_3d_smoothing = smooth_3D_trajectory(trajectory_3d)
    timing_results['3D平滑處理'] = time.perf_counter() - start
    print(f"-- 3D平滑處理完成，耗時：{timing_results['3D平滑處理']:.4f} 秒")

    create_3d_plots(trajectory_3d_smoothing)

    # 計算總時間
    total_time = time.perf_counter() - start_total

    # 輸出時間統計摘要
    print('\n' + '-'*50)
    print("執行時間統計摘要：")
    print('-'*50)
    for step, time_taken in timing_results.items():
        print(f"{step:.<30} {time_taken:>10.4f} 秒")
    print('-'*50)
    print(f"{'總執行時間':.<30} {total_time:>10.4f} 秒")
    print('='*50)

    return total_time

if __name__ == "__main__":
    yolo_pose_model = YOLO('model/yolov8n-pose.pt')
    yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
    video_side = 'trajectory/嘉洋__trajectory/trajectory__3/嘉洋__3_side.mp4'
    video_45 = 'trajectory/嘉洋__trajectory/trajectory__3/嘉洋__3_45.mp4'
    total_time = process_trajectory(video_side, video_45, yolo_pose_model, yolo_tennis_ball_model)