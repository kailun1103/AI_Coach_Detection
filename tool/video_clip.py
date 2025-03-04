import cv2
import numpy as np
from tqdm import tqdm

def process_video_fast(input_path, output_path, speed_factor=1.0):
    # 讀取影片
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("無法開啟影片檔案!")
        return
    
    # 獲取影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"總幀數: {total_frames}")
    print(f"FPS: {original_fps}")
    print(f"解析度: {width}x{height}")
    
    # 讓使用者輸入要保存的範圍
    start_frame = int(input("請輸入起始幀數 (從0開始): "))
    end_frame = int(input("請輸入結束幀數: "))
    
    if start_frame < 0 or end_frame >= total_frames or start_frame >= end_frame:
        print("輸入的幀數範圍無效!")
        return
    
    # 計算實際要處理的幀數
    frames_to_process = end_frame - start_frame + 1
    
    # 使用 H.264 編碼器
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 改用 avc1 編碼器
    
    # 設定加速後的 FPS
    output_fps = int(original_fps * speed_factor)
    
    # 創建輸出視頻寫入器
    out = None
    
    # 設定當前幀位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    try:
        # 使用 tqdm 顯示進度條
        with tqdm(total=frames_to_process, desc="處理進度") as pbar:
            frames_processed = 0
            
            while frames_processed < frames_to_process:
                ret, frame = cap.read()
                
                if not ret:
                    print(f"\n在幀 {frames_processed + start_frame} 處讀取失敗")
                    break
                
                # 延遲創建輸出視頻寫入器，確保有有效的幀
                if out is None:
                    out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame.shape[1], frame.shape[0]))
                    if not out.isOpened():
                        print("\n無法創建輸出視頻檔案!")
                        return
                
                out.write(frame)
                frames_processed += 1
                pbar.update(1)
                
    except Exception as e:
        print(f"\n處理時發生錯誤: {str(e)}")
        
    finally:
        # 釋放資源
        if out is not None:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
        
    if frames_processed == frames_to_process:
        print("\n處理完成!")
    else:
        print(f"\n只完成了 {frames_processed}/{frames_to_process} 幀的處理")

# 使用範例
input_video = '謝老師__3_side.mp4'
output_video = "謝老師__3_side_temp.mp4"  # 改用不同的輸出檔名
speed_factor = 1.0  # 保持原速
process_video_fast(input_video, output_video, speed_factor)