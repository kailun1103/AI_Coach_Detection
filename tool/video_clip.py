import cv2
import numpy as np
from tqdm import tqdm

def process_video_fast(input_path, output_path, speed_factor=1.0):
    # 讀取影片
    cap = cv2.VideoCapture(input_path)
    
    # 獲取影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"總幀數: {total_frames}")
    
    # 讓使用者輸入要保存的範圍
    start_frame = int(input("請輸入起始幀數 (從0開始): "))
    end_frame = int(input("請輸入結束幀數: "))
    
    # 計算實際要處理的幀數
    frames_to_process = end_frame - start_frame + 1
    
    # 使用 MP4V 編碼器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 設定加速後的 FPS
    output_fps = int(original_fps * speed_factor)
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    # 設定當前幀位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 使用更大的批量讀取來提升效能
    batch_size = 64  # 增加批量大小以提高效能
    
    # 使用 tqdm 顯示進度條
    with tqdm(total=frames_to_process, desc="處理進度") as pbar:
        frames_processed = 0
        while frames_processed < frames_to_process:
            frames = []
            # 批量讀取幀
            for _ in range(min(batch_size, frames_to_process - frames_processed)):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if not frames:
                break
                
            # 批量寫入幀
            for frame in frames:
                out.write(frame)
            
            frames_processed += len(frames)
            pbar.update(len(frames))
    
    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("完成!")

# 使用範例
input_video = "pro_1_1_side_完成.mp4"
output_video = "pro_1_1_side_temp.mp4"
speed_factor = 1.0  # 保持原速
process_video_fast(input_video, output_video, speed_factor)