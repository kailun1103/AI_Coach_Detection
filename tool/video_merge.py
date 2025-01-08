import cv2
import numpy as np
from tqdm import tqdm

def combine_videos_fast(top_video_path, bottom_video_path, output_path):
    # 開啟影片
    cap1 = cv2.VideoCapture(top_video_path)
    cap2 = cv2.VideoCapture(bottom_video_path)
    
    # 取得影片資訊
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    total_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), 
                      int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    print(f"總幀數: {total_frames}")
    
    # 設定輸出影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height*2))
    
    # 使用批量處理來提升效能
    batch_size = 64  # 可以根據可用記憶體調整這個數值
    
    # 使用 tqdm 顯示進度條
    with tqdm(total=total_frames, desc="合併進度") as pbar:
        frames_processed = 0
        
        while frames_processed < total_frames:
            top_frames = []
            bottom_frames = []
            
            # 批量讀取幀
            for _ in range(min(batch_size, total_frames - frames_processed)):
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    break
                    
                top_frames.append(frame1)
                bottom_frames.append(frame2)
            
            if not top_frames or not bottom_frames:
                break
            
            # 批量處理和寫入
            for top_frame, bottom_frame in zip(top_frames, bottom_frames):
                combined_frame = np.vstack((top_frame, bottom_frame))
                out.write(combined_frame)
            
            frames_processed += len(top_frames)
            pbar.update(len(top_frames))
    
    # 釋放資源
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    print("完成!")

# 使用範例
top_video = "leftBackhand_45.mp4"
bottom_video = "leftBackhand_side.mp4"
output_video = "leftBackhand_merge.mp4"
combine_videos_fast(top_video, bottom_video, output_video)