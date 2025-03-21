"""
整合 2D/3D 軌跡分析、影片處理、軌跡同步、KNN 與 GPT 反饋生成的整體流程。
此程式會依序完成：
  1. 從側面與 45° 影片中提取 2D 軌跡。
  2. 對 2D 軌跡進行平滑、插值與擊球角度處理。
  3. 處理影片（前處理/物件偵測）。
  4. 同步處理後的影片。
  5. 合併同步後的影片。
  6. 同步不同角度的軌跡資料。
  7. 使用兩組 2D 軌跡與攝影機投影矩陣 (P1, P2) 計算 3D 軌跡。
  8. 對 3D 軌跡進行平滑處理。
  9. 擷取有效擊球範圍（根據 2D 軌跡判斷，並在 3D 軌跡中提取）。
 10. 以 KNN 模組對 3D 軌跡進行初步分析。
 11. 最後根據 KNN 分析與 3D 擊球範圍，生成 GPT 文字化反饋。

各步驟皆計算執行時間，最後輸出時間統計摘要。

請根據實際路徑與需求調整各模組與檔案名稱。
"""

import time
import numpy as np
from ultralytics import YOLO
import concurrent.futures  # 新增多執行緒支援

# 匯入各模組 (請確認路徑正確)
from trajectory_2D_output import analyze_trajectory
from trajector_2D_smoothing import smooth_2D_trajectory
from video_detection import process_video
from video_sync import synchronize_videos
from video_merge import combine_videos_ffmpeg
from trajector_2D_sync import sync_trajectories
from trajector_2D_capture_swing_range import find_range
from trajectory_3D_output import process_trajectories
from trajector_3D_smoothing import smooth_3D_trajectory
from trajector_3D_capture_swing_range import extract_frames
from trajectory_knn import analyze_trajectory as analyze_trajectory_knn
from trajectory_gpt_single_feedback import generate_feedback

def processing_trajectory(P1, P2, yolo_pose_model, yolo_tennis_ball_model, video_side, video_45, knn_dataset):
    # 用於紀錄各步驟執行時間
    timing_results = {}
    start_total = time.perf_counter()  # 總執行時間計時

    # ------------------------------
    # print("\n步驟1：分析2D軌跡中...")
    start = time.perf_counter()
    trajectory_side = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_side, 28)
    trajectory_45  = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_45, 28)
    timing_results['2D軌跡分析'] = time.perf_counter() - start
    # print(f"-- 分析2D軌跡完成，耗時：{timing_results['2D軌跡分析']:.4f} 秒")

    # ------------------------------
    # 步驟2：2D 軌跡平滑/插值/擊球角度處理
    # ------------------------------
    # print("\n步驟2：進行2D軌跡平滑化/插值/擊球角度處理...")
    start = time.perf_counter()
    trajectory_side_smoothing = smooth_2D_trajectory(trajectory_side)
    trajectory_45_smoothing   = smooth_2D_trajectory(trajectory_45)
    timing_results['2D平滑處理'] = time.perf_counter() - start
    # print(f"-- 2D平滑處理完成，耗時：{timing_results['2D平滑處理']:.4f} 秒")

    # ------------------------------
    # 步驟3：影片處理
    # ------------------------------
    # print("\n步驟3：處理影片中...")
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 同時提交兩個視頻處理任務
        future_side = executor.submit(process_video, video_side)
        future_45 = executor.submit(process_video, video_45)
        
        # 獲取處理結果
        video_side_processed = future_side.result()
        video_45_processed = future_45.result()
    timing_results['影片處理'] = time.perf_counter() - start
    # print(f"-- 影片處理完成，耗時：{timing_results['影片處理']:.4f} 秒")

    # ------------------------------
    # 步驟4：影片同步
    # ------------------------------
    # print("\n步驟4：同步影片中...")
    start = time.perf_counter()
    synchronize_videos(video_side_processed, video_45_processed, 
                        trajectory_side_smoothing, trajectory_45_smoothing)
    timing_results['影片同步'] = time.perf_counter() - start
    # print(f"-- 影片同步完成，耗時：{timing_results['影片同步']:.4f} 秒")

    # ------------------------------
    # 步驟5：合併影片
    # ------------------------------
    # print("\n步驟5：合併影片中...")
    start = time.perf_counter()
    combine_videos_ffmpeg(video_45_processed, video_side_processed)
    timing_results['影片合併'] = time.perf_counter() - start
    # print(f"-- 影片合併完成，耗時：{timing_results['影片合併']:.4f} 秒")

    # ------------------------------
    # 步驟6：軌跡同步
    # ------------------------------
    # print("\n步驟6：同步軌跡中...")
    start = time.perf_counter()
    sync_trajectories(trajectory_side_smoothing, trajectory_45_smoothing)
    timing_results['軌跡同步'] = time.perf_counter() - start
    # print(f"-- 軌跡同步完成，耗時：{timing_results['軌跡同步']:.4f} 秒")

    # ------------------------------
    # 步驟7：3D 軌跡分析
    # ------------------------------
    # print("\n步驟7：計算3D軌跡中...")
    start = time.perf_counter()
    trajectory_3d = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)
    timing_results['3D軌跡分析'] = time.perf_counter() - start
    # print(f"-- 3D軌跡計算完成，耗時：{timing_results['3D軌跡分析']:.4f} 秒")

    # ------------------------------
    # 步驟8：3D 軌跡平滑處理
    # ------------------------------
    # print("\n步驟8：進行3D軌跡平滑處理中...")
    start = time.perf_counter()
    trajectory_3d_smoothing = smooth_3D_trajectory(trajectory_3d)
    timing_results['3D平滑處理'] = time.perf_counter() - start
    # print(f"-- 3D平滑處理完成，耗時：{timing_results['3D平滑處理']:.4f} 秒")

    # ------------------------------
    # 步驟9：有效擊球範圍判斷
    # ------------------------------
    # print("\n步驟9：判斷有效擊球範圍中...")
    start = time.perf_counter()
    start_frame, end_frame = find_range(trajectory_side_smoothing)
    trajectory_3d_swing_range = extract_frames(trajectory_3d_smoothing, start_frame, end_frame)
    timing_results['有效擊球範圍判斷'] = time.perf_counter() - start
    # print(f"-- 有效擊球範圍判斷完成，耗時：{timing_results['有效擊球範圍判斷']:.4f} 秒")

    # ------------------------------
    # 步驟10：KNN 分析
    # ------------------------------
    # print("\n步驟10：KNN 分析中...")
    start = time.perf_counter()
    trajectory_knn_suggestion = analyze_trajectory_knn(knn_dataset, trajectory_3d_smoothing)
    timing_results['KNN 分析'] = time.perf_counter() - start
    # print(f"-- KNN 分析完成，耗時：{timing_results['KNN 分析']:.4f} 秒")

    # ------------------------------
    # 步驟11：GPT 反饋生成
    # ------------------------------
    # print("\n步驟11：生成 GPT 反饋中...")
    start = time.perf_counter()
    trajectory_gpt_suggestion = generate_feedback(trajectory_3d_swing_range, trajectory_knn_suggestion)
    timing_results['GPT 反饋生成'] = time.perf_counter() - start
    # print(f"-- GPT 反饋生成完成，耗時：{timing_results['GPT 反饋生成']:.4f} 秒")

    # ------------------------------
    # 統計總執行時間並輸出時間摘要
    # ------------------------------
    total_time = time.perf_counter() - start_total
    print('\n' + '-' * 50)
    print("執行時間統計摘要：")
    print(f'處理影片: {video_side}')
    print('-' * 50)
    for step, t in timing_results.items():
        print(f"{step:.<30} {t:>10.4f} 秒")
    print('-' * 50)
    print(f"{'總執行時間':.<30} {total_time:>10.4f} 秒")
    print('=' * 50)

    return True

if __name__ == "__main__":

    # 碩士實驗室投影矩陣
    P1 = np.array([
        [  877.037008,     0.000000,   956.954783,     0.000000],
        [    0.000000,   879.565925,   564.021385,     0.000000],
        [    0.000000,     0.000000,     1.000000,     0.000000],
    ])

    P2 = np.array([
        [  408.666240,    -7.066100,  1265.246736, -264697.889698],
        [ -232.265915,   870.289013,   512.645370, 42861.701021],
        [   -0.400331,    -0.014736,     0.916252,    76.895470],
    ])

    # outdoor_11_26投影矩陣
    # P1 = np.array([
    #     [ 4930.662905,     0.000000,  1779.941295,     0.000000],
    #     [    0.000000,  3868.767102,  1001.404479,     0.000000],
    #     [    0.000000,     0.000000,     1.000000,     0.000000],
    # ])

    # P2 = np.array([
    #     [-1094.792294, -2221.390563,  5064.585259, -36565395.005422],
    #     [-1538.153919,  4167.785827,  2858.175091, -22603156.999564],
    #     [   -0.934159,    -0.007813,     0.356772,  2984.789713],
    # ])

    knn_dataset = 'knn_dataset.json'

    yolo_pose_model = YOLO('model/yolov8n-pose.pt')
    yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')

    yolo_pose_model.model.to('cuda')
    yolo_tennis_ball_model.model.to('cuda')



    video_side = f'trajectory/testing_123/testing__side.mp4'
    video_45 = f'trajectory/testing_123/testing__45.mp4'
    process_status = processing_trajectory(P1, P2, yolo_pose_model, yolo_tennis_ball_model, video_side, video_45, knn_dataset)
    print(process_status)