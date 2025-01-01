import cv2
import numpy as np

def combine_videos(top_video_path, bottom_video_path, output_path):
   # 開啟影片
   cap1 = cv2.VideoCapture(top_video_path)
   cap2 = cv2.VideoCapture(bottom_video_path)
   
   # 取得影片資訊
   width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps = int(cap1.get(cv2.CAP_PROP_FPS))
   
   # 設定輸出影片
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter(output_path, fourcc, fps, (width, height*2))
   
   while True:
       ret1, frame1 = cap1.read()
       ret2, frame2 = cap2.read()
       
       if not ret1 or not ret2:
           break
           
       # 上下合併
       combined_frame = np.vstack((frame1, frame2))
       out.write(combined_frame)
   
   # 釋放資源
   cap1.release()
   cap2.release()
   out.release()

# 使用範例
top_video = "left_ball_trail_slow.mp4"
bottom_video = "leftFront_ball_trail_slow.mp4" 
output_video = "leftBackhand_combined.mp4"
combine_videos(top_video, bottom_video, output_video)