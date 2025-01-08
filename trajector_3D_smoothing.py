import json
import numpy as np
from scipy.signal import savgol_filter
import time

def smooth_3D_trajectory(input_path, window_length=15, polyorder=3, angle_threshold=30):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # 平滑手腕軌跡
    coords = {axis: savgol_filter([frame['left_wrist'][axis] for frame in data], 
                                 window_length, polyorder)
            for axis in ['x', 'y', 'z']}

    # 獲取有效的球軌跡幀
    ball_frames = [(i, frame['tennis_ball']) for i, frame in enumerate(data) 
                   if not any(frame['tennis_ball'][axis] is None for axis in ['x','y','z'])]
    
    if ball_frames:
        start_frame, end_frame = ball_frames[0][0], ball_frames[-1][0]
        
        # 獲取有效點
        valid_points = [[data[i]['tennis_ball']['x'], 
                        data[i]['tennis_ball']['y'],
                        data[i]['tennis_ball']['z'], i]
                       for i in range(start_frame, end_frame + 1)
                       if not any(data[i]['tennis_ball'][axis] is None for axis in ['x','y','z'])]
        
        # 找到角度點
        angle_points = []
        for i in range(1, len(valid_points) - 1):
            frame_idx = valid_points[i][3]
            # 檢查是否為擊球點
            if data[frame_idx].get('tennis_hit', False):
                angle_points.append(frame_idx)
                continue
                
            p1, p2, p3 = map(lambda x: np.array(x[:3]), 
                            [valid_points[i-1], valid_points[i], valid_points[i+1]])
            v1, v2 = p1 - p2, p3 - p2
            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / 
                             (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
            
            if angle < angle_threshold:
                angle_points.append(frame_idx)
        
        # 創建和平滑段落
        segments = []
        angle_points = sorted(angle_points)
        
        # 找到擊球點
        hit_frames = [i for i, frame in enumerate(data) if frame.get('tennis_hit', False)]
        
        if hit_frames:
            # 根據擊球點分割段落
            current_start = start_frame
            for hit_frame in hit_frames:
                if hit_frame > current_start:
                    segments.append((current_start, hit_frame-1))
                    segments.append((hit_frame, hit_frame))  # 擊球點單獨作為一個段落
                    current_start = hit_frame + 1
            if current_start < end_frame:
                segments.append((current_start, end_frame))
        else:
            # 如果沒有擊球點，使用原來的邏輯
            segments = [(last := start_frame, pt) for pt in angle_points]
            segments.append((angle_points[-1] if angle_points else start_frame, end_frame))
        
        # 對每個段落進行平滑處理
        for seg_start, seg_end in segments:
            # 如果是擊球點段落，跳過平滑處理
            if seg_start == seg_end and data[seg_start].get('tennis_hit', False):
                continue
                
            ball_coords = {axis: [] for axis in ['x','y','z']}
            frames = []
            
            for i in range(seg_start, seg_end + 1):
                ball = data[i]['tennis_ball']
                if not any(ball[axis] is None for axis in ['x','y','z']):
                    for axis in ['x','y','z']:
                        ball_coords[axis].append(ball[axis])
                    frames.append(i)
            
            if len(frames) > window_length:
                for axis in ['x','y','z']:
                    interp = np.interp(range(seg_start, seg_end + 1), frames, ball_coords[axis])
                    smoothed = savgol_filter(interp, window_length, polyorder)
                    
                    for i, frame_idx in enumerate(range(seg_start, seg_end + 1)):
                        if not data[frame_idx].get('tennis_hit', False):  # 只有非擊球點才更新
                            data[frame_idx]['tennis_ball'][axis] = float(smoothed[i])
    
    # 更新軌跡並保留 tennis_hit
    smoothed_data = [{
        'frame': data[i]['frame'],
        'left_wrist': {axis: float(coords[axis][i]) for axis in ['x','y','z']},
        'tennis_ball': data[i]['tennis_ball'],
        'tennis_hit': data[i].get('tennis_hit', False)
    } for i in range(len(data))]
    
    output_path = input_path.replace('.json','_smoothed.json')
    with open(output_path, 'w') as f:
        json.dump(smoothed_data, f, indent=2)
        
    return output_path

if __name__ == "__main__":
    start = time.time()
    input_path = "leftBackhand_3D_trajectory_with_hits.json"
    output_path = smooth_3D_trajectory(input_path)
    print(f"Execution time: {time.time() - start:.4f}s")