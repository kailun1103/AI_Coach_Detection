import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps
    cap.release()
    return total_frames, fps, duration

def process_video(input_path, output_path, start_frame, frames_to_process, dimensions):
    cap = cv2.VideoCapture(input_path)
    width, height = dimensions
    
    # 使用 H.264 編碼器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    
    # 設置讀取緩衝區大小
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1024)
    
    # 跳到起始幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 批次讀取和寫入
    batch_size = 32
    frames = []
    
    for i in range(0, frames_to_process, batch_size):
        batch_frames = min(batch_size, frames_to_process - i)
        for _ in range(batch_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        # 批次寫入
        for frame in frames:
            out.write(frame)
        frames = []
    
    cap.release()
    out.release()

def synchronize_videos(input_path_1, input_path_2, hit_frame_1, hit_frame_2, trim_length=60):
    # 獲取影片資訊
    frames1, fps1, duration1 = get_video_info(input_path_1)
    frames2, fps2, duration2 = get_video_info(input_path_2)
    
    print("\n原始影片資訊:")
    print(f"影片 1: {frames1} 幀, {duration1:.2f} 秒")
    print(f"影片 2: {frames2} 幀, {duration2:.2f} 秒")
    
    print(f"\n擊球幀位置:")
    print(f"影片 1: 第 {hit_frame_1} 幀")
    print(f"影片 2: 第 {hit_frame_2} 幀")

    # 計算剪輯範圍
    max_frames_after = min(frames1 - hit_frame_1, frames2 - hit_frame_2)
    max_frames_before = min(hit_frame_1, hit_frame_2)
    
    frames_before = min(trim_length // 2, max_frames_before)
    frames_after = min(trim_length - frames_before, max_frames_after)
    
    start_frame_1 = hit_frame_1 - frames_before
    start_frame_2 = hit_frame_2 - frames_before
    frames_to_process = frames_before + frames_after

    print(f"\n剪輯資訊:")
    print(f"影片 1: 從第 {start_frame_1} 幀到第 {start_frame_1 + frames_to_process} 幀")
    print(f"影片 2: 從第 {start_frame_2} 幀到第 {start_frame_2 + frames_to_process} 幀")

    # 獲取影片尺寸
    cap1 = cv2.VideoCapture(input_path_1)
    cap2 = cv2.VideoCapture(input_path_2)
    dimensions1 = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                  int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    dimensions2 = (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                  int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap1.release()
    cap2.release()

    # 設定輸出檔案名稱
    output_path_1 = input_path_1.replace('.mp4', '_sync.mp4')
    output_path_2 = input_path_2.replace('.mp4', '_sync.mp4')

    # 使用線程池並行處理兩個影片
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_video, input_path_1, output_path_1, 
                          start_frame_1, frames_to_process, dimensions1),
            executor.submit(process_video, input_path_2, output_path_2, 
                          start_frame_2, frames_to_process, dimensions2)
        ]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"處理過程中發生錯誤: {str(e)}")

    final_duration = frames_to_process / fps1
    print(f"\n最終影片資訊:")
    print(f"兩個影片都是 {frames_to_process} 幀, {final_duration:.2f} 秒")
    print(f"\n同步完成! 輸出檔案:")
    print(f"影片 1: {output_path_1}")
    print(f"影片 2: {output_path_2}")
    
    return output_path_1, output_path_2

if __name__ == "__main__":
    start_time = time.time()
    
    # 輸入影片路徑和打擊點
    input_video_1 = input("請輸入第一個影片的路徑: ")
    input_video_2 = input("請輸入第二個影片的路徑: ")
    hit_frame_1 = int(input("請輸入第一個影片的打擊幀數: "))
    hit_frame_2 = int(input("請輸入第二個影片的打擊幀數: "))
    
    print("\n開始執行影片同步...")
    output_path_1, output_path_2 = synchronize_videos(
        input_video_1, input_video_2, 
        hit_frame_1, hit_frame_2
    )
    
    print(f"\n執行時間: {time.time() - start_time:.4f}秒")