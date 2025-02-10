import cv2
import numpy as np
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def read_frames_batch(cap, batch_size):
    """批次讀取影格"""
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def combine_videos_cpu(top_video_path, bottom_video_path):
    # 開啟影片
    cap1 = cv2.VideoCapture(top_video_path)
    cap2 = cv2.VideoCapture(bottom_video_path)
    
    # 設置較大的緩衝區以提升讀取速度
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1024)
    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1024)
    
    # 獲取影片資訊
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    total_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), 
                      int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # print(f"總幀數: {total_frames}")
    
    # 使用 mp4v 編碼器
    output_path = 'pro_1_1.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height*2))
    
    # 設定批次大小
    batch_size = 32
    
    # 預分配記憶體給合併後的影格
    combined_frame = np.zeros((height*2, width, 3), dtype=np.uint8)
    
    # 使用進度條
    with tqdm(total=total_frames, desc="合併進度") as pbar:
        frames_processed = 0
        
        while frames_processed < total_frames:
            # 讀取批次影格
            top_frames = read_frames_batch(cap1, batch_size)
            bottom_frames = read_frames_batch(cap2, batch_size)
            
            if not top_frames or not bottom_frames:
                break
            
            # 確保兩個批次的大小相同
            min_frames = min(len(top_frames), len(bottom_frames))
            
            # 批次處理影格
            for i in range(min_frames):
                # 直接在預分配的陣列中寫入數據
                combined_frame[:height] = top_frames[i]
                combined_frame[height:] = bottom_frames[i]
                out.write(combined_frame)
            
            frames_processed += min_frames
            pbar.update(min_frames)
    
    # 釋放資源
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_time = time.time()
    
    top_video = "pro_1_1_45_temp.mp4"
    bottom_video = "pro_1_1_side_temp.mp4"
    
    print("開始合併影片...")
    combine_videos_cpu(top_video, bottom_video)
    
    print(f"執行時間: {time.time() - start_time:.4f}秒")