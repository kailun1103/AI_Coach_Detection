from pymediainfo import MediaInfo
import cv2
import os

def get_relative_frame_number(start_tc, target_tc, fps=59.94):
    """計算相對幀數"""
    def tc_to_frames(tc):
        hours, minutes, seconds, frames = map(int, tc.split(':'))
        return hours * 3600 * fps + minutes * 60 * fps + seconds * fps + frames

    start_frames = tc_to_frames(start_tc)
    target_frames = tc_to_frames(target_tc)
    return int(target_frames - start_frames)

def analyze_and_process_videos(video1_path, video2_path, output1_path, output2_path):
    # 檢查輸入文件
    if not os.path.exists(video1_path) or not os.path.exists(video2_path):
        print("錯誤: 找不到輸入文件")
        return

    # 分析影片時間碼
    info1 = MediaInfo.parse(video1_path)
    info2 = MediaInfo.parse(video2_path)
    
    tc1 = None
    tc2 = None
    
    for track in info1.tracks:
        if track.track_type == "Other" and track.format == "QuickTime TC":
            tc1 = {
                "start": track.time_code_of_first_frame,
                "end": track.time_code_of_last_frame,
                "frames": track.frame_count
            }
    
    for track in info2.tracks:
        if track.track_type == "Other" and track.format == "QuickTime TC":
            tc2 = {
                "start": track.time_code_of_first_frame,
                "end": track.time_code_of_last_frame,
                "frames": track.frame_count
            }
    
    if not tc1 or not tc2:
        print("錯誤: 無法獲取時間碼信息")
        return

    # 計算同步點
    sync_start = max(tc1['start'], tc2['start'])
    sync_end = min(tc1['end'], tc2['end'])
    
    print(f"\n同步剪輯點:")
    print(f"開始時間: {sync_start}")
    print(f"結束時間: {sync_end}")
    
    # 計算每個影片的相對幀數
    start_frame1 = get_relative_frame_number(tc1['start'], sync_start)
    end_frame1 = get_relative_frame_number(tc1['start'], sync_end)
    start_frame2 = get_relative_frame_number(tc2['start'], sync_start)
    end_frame2 = get_relative_frame_number(tc2['start'], sync_end)
    
    print(f"\n影片1相對幀數:")
    print(f"開始幀: {start_frame1}")
    print(f"結束幀: {end_frame1}")
    print(f"\n影片2相對幀數:")
    print(f"開始幀: {start_frame2}")
    print(f"結束幀: {end_frame2}")

    def process_video(input_path, output_path, start_frame, end_frame):
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"錯誤: 無法打開影片 {input_path}")
                return False

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"\n影片資訊 {input_path}:")
            print(f"FPS: {fps}")
            print(f"解析度: {width}x{height}")
            print(f"總幀數: {total_frames}")
            print(f"預計處理幀數: {end_frame - start_frame + 1}")

            if start_frame >= total_frames or end_frame >= total_frames:
                print(f"錯誤: 幀數超出範圍 {input_path}")
                return False

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"錯誤: 無法創建輸出文件 {output_path}")
                return False

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_processed = 0
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    print(f"錯誤: 在幀 {frame_idx} 讀取失敗")
                    break
                
                out.write(frame)
                frames_processed += 1
                
                if frame_idx % 100 == 0:
                    print(f"處理進度 {input_path}: {frame_idx}/{end_frame}")

            print(f"成功處理的幀數: {frames_processed}")
            
            cap.release()
            out.release()
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"{input_path} 處理完成!")
                return True
            else:
                print(f"錯誤: 輸出文件 {output_path} 可能有問題")
                return False

        except Exception as e:
            print(f"處理 {input_path} 時發生錯誤: {str(e)}")
            return False

    # 處理兩個影片
    print("\n開始處理第一個影片...")
    success1 = process_video(video1_path, output1_path, start_frame1, end_frame1)
    
    print("\n開始處理第二個影片...")
    success2 = process_video(video2_path, output2_path, start_frame2, end_frame2)

    if success1 and success2:
        print("\n兩個影片都成功處理完成!")
    else:
        print("\n處理過程中發生錯誤，請檢查輸出信息。")

# 使用示例
video1_path = "test1.mp4"
video2_path = "test2.mp4"
output1_path = "test1_synced.mp4"
output2_path = "test2_synced.mp4"

analyze_and_process_videos(video1_path, video2_path, output1_path, output2_path)