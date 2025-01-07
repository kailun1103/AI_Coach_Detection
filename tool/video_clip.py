import cv2

def process_video(input_path, output_path):
    # 讀取影片
    cap = cv2.VideoCapture(input_path)
    
    # 獲取影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"總幀數: {total_frames}")
    
    # 讓使用者輸入要保存的範圍
    start_frame = int(input("請輸入起始幀數 (從0開始): "))
    end_frame = int(input("請輸入結束幀數: "))
    
    # 設定輸出影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 設定當前幀位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 讀取並保存指定範圍的幀
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        
        # 顯示進度
        if frame_idx % 100 == 0:
            print(f"處理進度: {frame_idx}/{end_frame}")
    
    # 釋放資源
    cap.release()
    out.release()
    print("完成!")

# 使用範例
input_video = "test1_correct.mp4"  # 輸入影片路徑
output_video = "test1_correct.mp4"  # 輸出影片路徑
process_video(input_video, output_video)