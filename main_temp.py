import time
import numpy as np 
from ultralytics import YOLO
from trajectory_2D_output_temp import analyze_trajectory
from trajector_2D_smoothing import smooth_2D_trajectory
from trajectory_3D_output import process_trajectories
from trajector_3D_smoothing import smooth_3D_trajectory
from trajector_2D_sync import sync_trajectories
from drawing_3D_plotly import create_3d_plots
from video_detection import process_video
from video_sync import synchronize_videos
from video_merge import combine_videos_cpu
from trajectory_2D_output_video_detection import process_tennis_video
import concurrent.futures

def analyze_videos_parallel(yolo_pose_model, yolo_tennis_ball_model, video_side, video_45):
    """
    並行執行兩個影片的分析任務
    
    Args:
        yolo_pose_model: YOLO 姿態偵測模型
        yolo_tennis_ball_model: YOLO 網球偵測模型
        video_side: 側面影片路徑
        video_45: 45度影片路徑
        
    Returns:
        tuple: (側面影片軌跡, 45度影片軌跡)
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_side = executor.submit(
            analyze_trajectory, 
            yolo_pose_model, 
            yolo_tennis_ball_model, 
            video_side
        )
        future_45 = executor.submit(
            analyze_trajectory, 
            yolo_pose_model, 
            yolo_tennis_ball_model, 
            video_45
        )
        
        trajectory_side = future_side.result()
        trajectory_45 = future_45.result()
        
    return trajectory_side, trajectory_45

def start_process_videos_parallel(video_side, video_45):
    """
    開始並行處理兩個影片，但不等待完成
    
    Args:
        video_side: 側面影片路徑
        video_45: 45度影片路徑
    
    Returns:
        executor: ThreadPoolExecutor 實例
        futures: 包含兩個任務的 future 列表
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    future_side = executor.submit(process_video, video_side)
    future_45 = executor.submit(process_video, video_45)
    
    return executor, [future_side, future_45]

def main():
    # 開始計算總執行時間
    start_total = time.perf_counter()
    timing_results = {}

    # 輸入影片
    video_side = 'pro_1_1_side_temp.mp4'
    video_45 = 'pro_1_1_45_temp.mp4'

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

    print("步驟1：載入模型中...")
    start = time.perf_counter()
    yolo_pose_model = YOLO('model/yolov8n-pose.pt')
    yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
    timing_results['模型載入'] = time.perf_counter() - start
    print(f"-- 模型載入完成，耗時：{timing_results['模型載入']:.4f} 秒")

    print("\n步驟2：開始影片前處理（背景執行）...")
    start_video_process = time.perf_counter()
    video_executor, video_futures = start_process_videos_parallel(video_side, video_45)
    print("-- 影片前處理已在背景開始執行")
    
    print("\n步驟3：分析2D軌跡中...")
    start = time.perf_counter()
    trajectory_side, trajectory_45 = analyze_videos_parallel(
        yolo_pose_model, 
        yolo_tennis_ball_model, 
        video_side, 
        video_45
    )
    timing_results['2D軌跡分析'] = time.perf_counter() - start
    print(f"-- 2D軌跡分析完成，耗時：{timing_results['2D軌跡分析']:.4f} 秒")

    print("\n步驟4：進行2D軌跡平滑化處理...")
    start = time.perf_counter()
    trajectory_side_smoothing = smooth_2D_trajectory(trajectory_side)
    trajectory_45_smoothing = smooth_2D_trajectory(trajectory_45)
    timing_results['2D平滑處理'] = time.perf_counter() - start
    print(f"-- 2D平滑處理完成，耗時：{timing_results['2D平滑處理']:.4f} 秒")

    print("\n步驟5：同步軌跡中...")
    start = time.perf_counter()
    sync_trajectories(trajectory_side_smoothing, trajectory_45_smoothing)
    timing_results['軌跡同步'] = time.perf_counter() - start
    print(f"-- 軌跡同步完成，耗時：{timing_results['軌跡同步']:.4f} 秒")

    print("\n步驟6：計算3D軌跡中...")
    start = time.perf_counter()
    trajectory_3d = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)
    timing_results['3D軌跡分析'] = time.perf_counter() - start
    print(f"-- 3D軌跡計算完成，耗時：{timing_results['3D軌跡分析']:.4f} 秒")

    print("\n步驟7：進行3D軌跡平滑處理中...")
    start = time.perf_counter()
    trajectory_3d_smoothing = smooth_3D_trajectory(trajectory_3d)
    timing_results['3D平滑處理'] = time.perf_counter() - start
    print(f"-- 3D平滑處理完成，耗時：{timing_results['3D平滑處理']:.4f} 秒")

    print("\n步驟8：同步影片中...")
    start = time.perf_counter()
    output_path_1, output_path_2 = synchronize_videos(str(video_side), video_45, 
                                                     trajectory_side_smoothing, 
                                                     trajectory_45_smoothing)
    timing_results['影片同步'] = time.perf_counter() - start
    print(f"-- 影片同步完成，耗時：{timing_results['影片同步']:.4f} 秒")

    print("\n等待影片前處理完成...")
    concurrent.futures.wait(video_futures)
    timing_results['影片處理'] = time.perf_counter() - start_video_process
    video_executor.shutdown()
    print(f"-- 影片前處理完成，耗時：{timing_results['影片處理']:.4f} 秒")

    print("\n步驟9：合併影片中...")
    start = time.perf_counter()
    combine_videos_cpu(output_path_1, output_path_2)
    timing_results['影片合併'] = time.perf_counter() - start
    print(f"-- 影片合併完成，耗時：{timing_results['影片合併']:.4f} 秒")

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

    return trajectory_3d_smoothing

if __name__ == "__main__":
    main()