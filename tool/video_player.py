import cv2

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("無法開啟影片檔案")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    speed = 0.1
    is_playing = True
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = width // 6
    new_height = height // 6
    
    # 讀取第一幀
    ret, frame = cap.read()
    
    while True:
        if is_playing:  # 播放狀態
            ret, frame = cap.read()
            if not ret:
                print("影片播放完畢")
                break
        
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # 顯示控制說明和當前狀態
        status = "Playing" if is_playing else "Paused"
        text = f"Frame: {current_frame} Speed: {speed:.1f}x Status: {status}"
        cv2.putText(resized_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        cv2.imshow('Video Player', resized_frame)
        
        # 等待按鍵輸入
        key = cv2.waitKey(int(1000/(fps*speed)) if is_playing else 0) & 0xFF
        
        # 按鍵控制
        if key == ord('q'):  # 退出
            break
        elif key == ord(' '):  # 空白鍵：播放/暫停
            is_playing = not is_playing
        elif key == ord('+') or key == ord('='):  # 加速
            speed = min(speed + 0.1, 2.0)
        elif key == ord('-'):  # 減速
            speed = max(speed - 0.1, 0.1)
        elif key == ord('r'):  # 重設速度
            speed = 1.0
        elif key == ord(',') or key == ord('<'):  # 倒退
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = max(0, current_pos - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
        elif key == ord('.') or key == ord('>'):  # 前進
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "0224_full_video.mp4"
    play_video(video_path)