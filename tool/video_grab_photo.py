import cv2
import os

def play_videos(video1_path, video2_path, base_output_folder='screenshots'):
    # 建立截圖儲存資料夾
    folder_45 = os.path.join(base_output_folder, '45')
    folder_side = os.path.join(base_output_folder, 'side')
    for folder in [folder_45, folder_side]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # 開啟兩個影片
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # 檢查影片是否成功開啟
    if not cap1.isOpened() or not cap2.isOpened():
        print("無法開啟影片檔案")
        return
    
    # 計數器用於檔名
    screenshot_counter = 0
    
    # 取得影片總幀數
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 目前播放的幀數
    current_frame1 = 0
    current_frame2 = 0
    
    print("控制說明：")
    print("'s': 截圖")
    print("'q': 結束播放")
    print("'a': 倒退 5 幀")
    print("'d': 前進 5 幀")
    print("'z': 倒退 1 幀")
    print("'c': 前進 1 幀")
    
    while True:
        # 讀取影片的每一幀
        ret1, frame1_original = cap1.read()
        ret2, frame2_original = cap2.read()
        
        # 如果影片播放完畢就重新播放
        if not ret1:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1_original = cap1.read()
            current_frame1 = 0
        if not ret2:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2_original = cap2.read()
            current_frame2 = 0
        
        # 製作低解析度版本用於顯示
        frame1_display = cv2.resize(frame1_original, (640, 480))
        frame2_display = cv2.resize(frame2_original, (640, 480))
        
        # 顯示影片和目前幀數
        cv2.imshow('Video 1', frame1_display)
        cv2.imshow('Video 2', frame2_display)
        
        # 顯示進度
        print(f"\r播放進度 - Video 1: {current_frame1}/{total_frames1}, Video 2: {current_frame2}/{total_frames2}", end="")
        
        # 檢查鍵盤輸入
        key = cv2.waitKey(1) & 0xFF
        
        # 按下 's' 鍵截圖
        if key == ord('s'):
            # 儲存原始解析度的截圖
            cv2.imwrite(os.path.join(folder_45, f'Indoor_{screenshot_counter}.JPG'), frame1_original)
            cv2.imwrite(os.path.join(folder_side, f'Indoor_{screenshot_counter}.JPG'), frame2_original)
            print(f"\n已儲存截圖: Indoor_{screenshot_counter}.JPG")
            screenshot_counter += 1
        
        # 影片控制
        elif key == ord('a'):  # 倒退 5 幀
            current_frame1 = max(0, current_frame1 - 5)
            current_frame2 = max(0, current_frame2 - 5)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, current_frame1)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, current_frame2)
        
        elif key == ord('d'):  # 前進 5 幀
            current_frame1 = min(total_frames1 - 1, current_frame1 + 5)
            current_frame2 = min(total_frames2 - 1, current_frame2 + 5)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, current_frame1)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, current_frame2)
        
        elif key == ord('z'):  # 倒退 1 幀
            current_frame1 = max(0, current_frame1 - 1)
            current_frame2 = max(0, current_frame2 - 1)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, current_frame1)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, current_frame2)
        
        elif key == ord('c'):  # 前進 1 幀
            current_frame1 = min(total_frames1 - 1, current_frame1 + 1)
            current_frame2 = min(total_frames2 - 1, current_frame2 + 1)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, current_frame1)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, current_frame2)
        
        # 按下 'q' 鍵結束程式
        elif key == ord('q'):
            break
        
        # 更新當前幀數
        current_frame1 = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))
        current_frame2 = int(cap2.get(cv2.CAP_PROP_POS_FRAMES))
    
    # 釋放資源
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 設定影片路徑
    video1_path = "synchronized_videos/0315_45_sync.mp4"
    video2_path = "synchronized_videos/0315_side_sync.mp4"
    play_videos(video1_path, video2_path)