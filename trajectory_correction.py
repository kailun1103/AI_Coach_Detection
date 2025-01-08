import json

def process_frames(input_file, start_frame, end_frame):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 篩選指定範圍的frame
    filtered_data = [frame for frame in data if start_frame <= frame['frame'] <= end_frame]
    
    # 重新編排frame編號
    for new_frame, frame_data in enumerate(filtered_data):
        frame_data['frame'] = new_frame
    
    # 直接儲存回原始檔案
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)


def main():
    # 設定檔案路徑
    file_path = 'leftBackhand_45_trajectory.json'

    start_frame = int(input("請輸入起始 frame: "))
    end_frame = int(input("請輸入結束 frame: "))
    
    # 處理資料並直接儲存
    frames_count = process_frames(file_path, start_frame, end_frame)
    
    if frames_count > 0:
        print(f"已成功處理資料並更新檔案 {file_path}")
        print(f"處理後的資料包含 {frames_count} 個 frames")
        

if __name__ == "__main__":
    main()