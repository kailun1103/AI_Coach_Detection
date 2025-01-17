import json
import numpy as np
from scipy.signal import savgol_filter
import time
import math

def calculate_angle(p1, p2, p3):
    """計算三點的夾角，返回角度值 (0-180)"""
    def distance(a, b):
        return math.sqrt((a["x"] - b["x"])**2 + (a["y"] - b["y"])**2)

    a = distance(p2, p3)
    b = distance(p1, p3)
    c = distance(p1, p2)

    if a * b == 0:
        return 0

    cos_value = (a**2 + b**2 - c**2) / (2 * a * b)
    cos_value = max(-1, min(1, cos_value))
    
    return math.degrees(math.acos(cos_value))

def smooth_2D_trajectory(input_file, window_length=15, polyorder=3, tennis_window_length=7, tennis_polyorder=2):
    """
    平滑所有關鍵點和網球軌跡，並根據最大角度判斷擊球點
    
    參數:
    input_file (str): 輸入 JSON 檔案路徑
    window_length (int): 關鍵點平滑窗口長度
    polyorder (int): 關鍵點平滑多項式階數
    tennis_window_length (int): 網球軌跡平滑窗口長度
    tennis_polyorder (int): 網球軌跡平滑多項式階數
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 為每個 frame 初始化新的欄位
    for frame in data:
        frame['tennis_ball_hit'] = False
        frame['tennis_ball_angle'] = 0.0
    
    # 定義需要平滑的關鍵點
    keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # 特殊處理網球軌跡
    first_valid = None
    last_valid = None
    for i, frame in enumerate(data):
        if frame['tennis_ball']['x'] is not None and frame['tennis_ball']['y'] is not None:
            if first_valid is None:
                first_valid = i
            last_valid = i
    
    if first_valid is not None and last_valid is not None:
        valid_range_data = data[first_valid:last_valid + 1]
        frames = np.array([frame['frame'] for frame in valid_range_data])
        
        tb_x = np.array([frame['tennis_ball']['x'] for frame in valid_range_data], dtype=float)
        tb_y = np.array([frame['tennis_ball']['y'] for frame in valid_range_data], dtype=float)
        
        # 插值處理
        valid_tb = ~np.isnan(tb_x)
        if np.any(~valid_tb):
            tb_x[~valid_tb] = np.interp(frames[~valid_tb], frames[valid_tb], tb_x[valid_tb])
            tb_y[~valid_tb] = np.interp(frames[~valid_tb], frames[valid_tb], tb_y[valid_tb])
        
        # 平滑處理
        if len(tb_x) > tennis_window_length:
            tb_x_smooth = savgol_filter(tb_x, tennis_window_length, tennis_polyorder)
            tb_y_smooth = savgol_filter(tb_y, tennis_window_length, tennis_polyorder)
            
            # 更新網球資料
            for i in range(len(valid_range_data)):
                data[i + first_valid]['tennis_ball'].update({
                    'x': float(tb_x_smooth[i]),
                    'y': float(tb_y_smooth[i])
                })

        # 計算夾角並找出最大角度
        angles = []
        for i in range(first_valid + 1, last_valid):
            prev_frame = data[i - 1]
            curr_frame = data[i]
            next_frame = data[i + 1]

            angle = calculate_angle(
                prev_frame['tennis_ball'],
                curr_frame['tennis_ball'],
                next_frame['tennis_ball']
            )
            
            angles.append((i, angle))
            
            # 記錄角度到每個 frame
            data[i]['tennis_ball_angle'] = float(angle)

        # 找出最大角度和對應的 frame
        if angles:
            max_frame_idx, max_angle = max(angles, key=lambda x: x[1])
            
            # 設定擊球點標記
            data[max_frame_idx]['tennis_ball_hit'] = True
            
            print(f"最大角度: {max_angle:.2f}度，出現在 frame {data[max_frame_idx]['frame']}")

    # 平滑所有關鍵點
    for keypoint in keypoints:
        # 提取座標
        x_coords = [frame[keypoint]['x'] for frame in data]
        y_coords = [frame[keypoint]['y'] for frame in data]
        
        # 檢查是否有足夠的有效點進行平滑
        valid_points = [i for i, (x, y) in enumerate(zip(x_coords, y_coords)) 
                       if x is not None and y is not None]
        
        if len(valid_points) > window_length:
            # 建立有效點遮罩
            x_array = np.array(x_coords)
            y_array = np.array(y_coords)
            
            # 用鄰近值替換 None
            for i in range(len(x_array)):
                if x_array[i] is None or y_array[i] is None:
                    valid_indices = np.where([x is not None and y is not None 
                                            for x, y in zip(x_array, y_array)])[0]
                    if len(valid_indices) > 0:
                        nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - i))]
                        x_array[i] = x_array[nearest_idx]
                        y_array[i] = y_array[nearest_idx]
            
            # 轉換為浮點數陣列
            x_array = x_array.astype(float)
            y_array = y_array.astype(float)
            
            # 使用 Savitzky-Golay filter 進行平滑
            smooth_x = savgol_filter(x_array, window_length, polyorder)
            smooth_y = savgol_filter(y_array, window_length, polyorder)
            
            # 更新資料
            for i, (sx, sy) in enumerate(zip(smooth_x, smooth_y)):
                if x_coords[i] is not None and y_coords[i] is not None:
                    data[i][keypoint].update({
                        'x': float(sx),
                        'y': float(sy)
                    })
    
    # 儲存結果
    output_file = input_file.replace(').json', '_smoothed).json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_file

if __name__ == "__main__":
    start_time = time.time()
    input_path = "temp/junior_side_trajectory.json"
    smoothed_data = smooth_2D_trajectory(input_path)
    print(f"Execution time: {time.time() - start_time:.4f}s")