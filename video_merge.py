import cv2
import numpy as np
import time
from tqdm import tqdm

def combine_videos_fast(top_video_path, bottom_video_path, output_path):
    cap1 = cv2.VideoCapture(top_video_path)
    cap2 = cv2.VideoCapture(bottom_video_path)
    
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    total_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), 
                      int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    print(f"總幀數: {total_frames}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height*2))
    
    batch_size = 64
    
    with tqdm(total=total_frames, desc="合併進度") as pbar:
        frames_processed = 0
        
        while frames_processed < total_frames:
            top_frames = []
            bottom_frames = []
            
            for _ in range(min(batch_size, total_frames - frames_processed)):
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    break
                    
                top_frames.append(frame1)
                bottom_frames.append(frame2)
            
            if not top_frames or not bottom_frames:
                break
            
            for top_frame, bottom_frame in zip(top_frames, bottom_frames):
                combined_frame = np.vstack((top_frame, bottom_frame))
                out.write(combined_frame)
            
            frames_processed += len(top_frames)
            pbar.update(len(top_frames))
    
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    print("完成!")

if __name__ == "__main__":
    start_time = time.time()
    
    top_video = "pro_1_1_45.mp4"
    bottom_video = "pro_1_1_side.mp4"
    output_video = "merge.mp4"
    
    print("開始合併影片...")
    combine_videos_fast(top_video, bottom_video, output_video)
    
    print(f"執行時間: {time.time() - start_time:.4f}秒")