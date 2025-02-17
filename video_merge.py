import os
import time

def combine_videos_ffmpeg(top_video, bottom_video):
    output_video = bottom_video.replace('_45_processed.mp4','_full_video.mp4')
    cmd = (
        f'ffmpeg -hwaccel cuda -i "{top_video}" -i "{bottom_video}" '
        f'-filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" '
        f'-c:v h264_nvenc -preset p7 -profile:v high444p -qp 0 -b:v 50000k '
        f'-rc constqp -pix_fmt yuv444p -threads 8 -bf 2 {output_video}'
    )
    
    os.system(cmd)  # 執行 FFmpeg 指令

if __name__ == "__main__":
    start_time = time.time()  # 記錄開始時間
    top_video = "張凱倫__2_45_processed.mp4"
    bottom_video = "張凱倫__2_side_processed.mp4"

    print("開始合併影片（超高畫質 + GPU 加速）...")
    combine_videos_ffmpeg(top_video, bottom_video)
    end_time = time.time()  # 記錄結束時間

    elapsed_time = end_time - start_time  # 計算執行時間

    print(f"處理時間: {elapsed_time:.2f} 秒")  # 顯示處理時間
