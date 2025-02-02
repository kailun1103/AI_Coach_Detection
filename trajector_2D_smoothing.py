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
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for frame in data:
        frame['tennis_ball_hit'] = False
    
    keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

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
        
        valid_tb = ~np.isnan(tb_x)
        if np.any(~valid_tb):
            tb_x[~valid_tb] = np.interp(frames[~valid_tb], frames[valid_tb], tb_x[valid_tb])
            tb_y[~valid_tb] = np.interp(frames[~valid_tb], frames[valid_tb], tb_y[valid_tb])
        
        if len(tb_x) > tennis_window_length:
            tb_x_smooth = savgol_filter(tb_x, tennis_window_length, tennis_polyorder)
            tb_y_smooth = savgol_filter(tb_y, tennis_window_length, tennis_polyorder)
            
            # Calculate tennis ball angles for all frames
            for i in range(len(valid_range_data)):
                if i > 0 and i < len(valid_range_data) - 1:
                    p1 = {"x": tb_x_smooth[i-1], "y": tb_y_smooth[i-1]}
                    p2 = {"x": tb_x_smooth[i], "y": tb_y_smooth[i]}
                    p3 = {"x": tb_x_smooth[i+1], "y": tb_y_smooth[i+1]}
                    angle = calculate_angle(p1, p2, p3)
                else:
                    angle = 0
                
                data[i + first_valid]['tennis_ball'].update({
                    'x': float(tb_x_smooth[i]),
                    'y': float(tb_y_smooth[i])
                })
                data[i + first_valid]['tennis_ball_angle'] = float(angle)

        # Find frame with minimum x coordinate
        min_x_idx = np.argmin(tb_x_smooth)
        hit_frame_idx = min_x_idx + first_valid
        data[hit_frame_idx]['tennis_ball_hit'] = True
        print(f"找到擊球點在 frame {data[hit_frame_idx]['frame']}")

    # Set tennis_ball_angle to 0 for frames outside valid range
    for i in range(len(data)):
        if i < first_valid or i > last_valid:
            data[i]['tennis_ball_angle'] = 0

    for keypoint in keypoints:
        x_coords = [frame[keypoint]['x'] for frame in data]
        y_coords = [frame[keypoint]['y'] for frame in data]
        
        valid_points = [i for i, (x, y) in enumerate(zip(x_coords, y_coords)) 
                       if x is not None and y is not None]
        
        if len(valid_points) > window_length:
            x_array = np.array(x_coords)
            y_array = np.array(y_coords)
            
            for i in range(len(x_array)):
                if x_array[i] is None or y_array[i] is None:
                    valid_indices = np.where([x is not None and y is not None 
                                            for x, y in zip(x_array, y_array)])[0]
                    if len(valid_indices) > 0:
                        nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - i))]
                        x_array[i] = x_array[nearest_idx]
                        y_array[i] = y_array[nearest_idx]
            
            x_array = x_array.astype(float)
            y_array = y_array.astype(float)
            
            smooth_x = savgol_filter(x_array, window_length, polyorder)
            smooth_y = savgol_filter(y_array, window_length, polyorder)
            
            for i, (sx, sy) in enumerate(zip(smooth_x, smooth_y)):
                if x_coords[i] is not None and y_coords[i] is not None:
                    data[i][keypoint].update({
                        'x': float(sx),
                        'y': float(sy)
                    })
    
    output_file = input_file.replace(').json', '_smoothed).json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_file

if __name__ == "__main__":
    start_time = time.time()
    input_path = "pro_45_3_13(2D_trajectory).json"
    smoothed_data = smooth_2D_trajectory(input_path)
    print(f"Execution time: {time.time() - start_time:.4f}s")