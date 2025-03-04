import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
import time

def reverse_video(input_path, output_path=None, speed_factor=1.0):
    """
    將視頻畫面水平翻轉（左右顛倒）並可選擇加速處理
    
    參數:
    input_path: 輸入視頻的路徑
    output_path: 輸出視頻的路徑，如果未提供則在同目錄下創建 "mirrored_" 前綴的文件
    speed_factor: 速度因子 (1.0 是原速, 2.0 是兩倍速, 0.5 是半速)
    """
    # 如果未提供輸出路徑，自動生成一個
    if output_path is None:
        filename = os.path.basename(input_path)
        directory = os.path.dirname(input_path)
        output_path = os.path.join(directory, f"mirrored_{filename}")
    
    print(f"開始處理視頻: {input_path}")
    print(f"輸出將保存到: {output_path}")
    
    # 打開視頻文件
    cap = cv2.VideoCapture(input_path)
    
    # 檢查視頻是否成功打開
    if not cap.isOpened():
        print("錯誤: 無法打開視頻文件")
        return
    
    # 獲取原始視頻的屬性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 考慮速度因子調整
    adjusted_fps = fps * speed_factor
    
    # 創建視頻寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, adjusted_fps, (width, height))
    
    print(f"視頻信息: {width}x{height}, {fps} FPS, 總幀數: {total_frames}")
    print(f"處理後 FPS: {adjusted_fps}")
    
    # 讀取每一幀並水平翻轉（左右顛倒）
    print("讀取和水平翻轉幀...")
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 水平翻轉（左右顛倒）
        mirrored_frame = cv2.flip(frame, 1)  # 1表示水平翻轉，0表示垂直翻轉，-1表示同時水平和垂直翻轉
        
        # 寫入翻轉後的幀
        out.write(mirrored_frame)
        pbar.update(1)
    
    pbar.close()
    
    # 釋放資源
    cap.release()
    out.release()
    
    print(f"視頻已成功水平翻轉（左右顛倒）並保存到 {output_path}")

def process_city_videos(directory, output_directory=None, speed_factor=2.0):
    """
    以最快速度處理目錄中的所有視頻
    
    參數:
    directory: 包含城市視頻的目錄
    output_directory: 輸出目錄，如果未提供則使用 input_directory/mirrored
    speed_factor: 速度因子，設為 2.0 表示兩倍速處理
    """
    # 如果未提供輸出目錄，創建一個默認目錄
    if output_directory is None:
        output_directory = os.path.join(directory, "mirrored")
    
    # 確保輸出目錄存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 獲取目錄中的所有視頻文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(directory, file))
    
    print(f"找到 {len(video_files)} 個視頻文件")
    
    # 處理每個視頻
    start_time = time.time()
    for i, video_file in enumerate(video_files):
        print(f"\n處理視頻 {i+1}/{len(video_files)}: {video_file}")
        output_file = os.path.join(output_directory, f"mirrored_{os.path.basename(video_file)}")
        reverse_video(video_file, output_file, speed_factor)
    
    total_time = time.time() - start_time
    print(f"\n所有視頻處理完成！總耗時: {total_time:.2f} 秒")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='將視頻水平翻轉（左右顛倒）並可選擇加速處理')
    parser.add_argument('input', help='輸入視頻文件或目錄')
    parser.add_argument('-o', '--output', help='輸出視頻文件或目錄')
    parser.add_argument('-s', '--speed', type=float, default=1.0, help='速度因子 (默認: 1.0)')
    parser.add_argument('-c', '--city', action='store_true', help='啟用城市處理模式 (批量處理目錄下的視頻)')
    
    args = parser.parse_args()
    
    if args.city:
        process_city_videos(args.input, args.output, args.speed)
    else:
        reverse_video(args.input, args.output, args.speed)