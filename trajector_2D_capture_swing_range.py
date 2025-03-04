import json
import time

def find_range(input_file):
    # 加載 JSON 數據
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
            
    # 只檢查右手腕（right_wrist）
    target_part = 'right_wrist'
    
    # 初始化變量以跟踪最小 x 和最大 x
    min_x_value = float('inf')
    max_x_value = float('-inf')
    min_x_frame = -1
    max_x_frame = -1
    
    # 遍歷所有幀，只檢查右手腕
    for i, frame_data in enumerate(data):
        if target_part in frame_data and isinstance(frame_data[target_part], dict):
            wrist_data = frame_data[target_part]
            
            # 檢查 x 值最小
            if wrist_data['x'] is not None and wrist_data['x'] < min_x_value:
                min_x_value = wrist_data['x']
                min_x_frame = i
            
            # 檢查 x 值最大
            if wrist_data['x'] is not None and wrist_data['x'] > max_x_value:
                max_x_value = wrist_data['x']
                max_x_frame = i
    
    # 檢查是否找到有效值
    if min_x_frame == -1 or max_x_frame == -1:
        print(f"錯誤：找不到右手腕的有效 x 值")
        return -1, -1
    
    # 定義範圍
    start_frame = min_x_frame  # 具有最小 x 的幀
    end_frame = max_x_frame    # 具有最大 x 的幀
    
    frame_count = end_frame - start_frame + 1
    
    return start_frame, end_frame

if __name__ == "__main__":
    input_file = "凱倫__2_side(2D_trajectory_smoothed).json"
    
    # 開始計時
    start_time = time.time()
    
    # 執行範圍查找
    start_frame, end_frame = find_range(input_file)
    
    # 結束計時
    end_time = time.time()
    execution_time = end_time - start_time
    
    if start_frame >= 0 and end_frame >= 0:
        print(f"已提取範圍：從幀 {start_frame} 到幀 {end_frame}")
        print(f"代碼執行時間：{execution_time:.4f} 秒")