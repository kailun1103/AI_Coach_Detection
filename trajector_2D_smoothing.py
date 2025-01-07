import json
import numpy as np
from scipy.signal import savgol_filter
import time

def smooth_2D_trajectory(input_file, window_length=15, polyorder=3, angle_threshold=30):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    ball_frames = [(i, frame['tennis_ball']) for i, frame in enumerate(data) 
                   if frame['tennis_ball']['x'] is not None]
    
    if ball_frames:
        start_frame, end_frame = ball_frames[0][0], ball_frames[-1][0]
        
        valid_points = [[data[i]['tennis_ball']['x'], 
                        data[i]['tennis_ball']['y'], i]
                       for i in range(start_frame, end_frame + 1)
                       if data[i]['tennis_ball']['x'] is not None]
        
        # Skip angle detection for tennis ball
        segments = [(start_frame, end_frame)]
        
        # Smooth each segment
        for seg_start, seg_end in segments:
            coords = [(data[i]['tennis_ball']['x'], 
                      data[i]['tennis_ball']['y'], i)
                     for i in range(seg_start, seg_end + 1)
                     if data[i]['tennis_ball']['x'] is not None]
            
            if len(coords) > window_length:
                x, y, frames = zip(*coords)
                x_interp = np.interp(range(seg_start, seg_end + 1), frames, x)
                y_interp = np.interp(range(seg_start, seg_end + 1), frames, y)
                
                # Keep original coordinates instead of smoothing
                for i, frame_idx in enumerate(range(seg_start, seg_end + 1)):
                    if data[frame_idx]['tennis_ball']['x'] is not None:
                        continue
                    data[frame_idx]['tennis_ball'].update({
                        'x': float(x_interp[i]),
                        'y': float(y_interp[i])
                    })
    
    # Smooth wrist trajectory as before
    wrist_x = [frame['left_wrist']['x'] for frame in data]
    wrist_y = [frame['left_wrist']['y'] for frame in data]
    smooth_x = savgol_filter(wrist_x, window_length, polyorder)
    smooth_y = savgol_filter(wrist_y, window_length, polyorder)
    
    for frame, sx, sy in zip(data, smooth_x, smooth_y):
        frame['left_wrist'].update({'x': float(sx), 'y': float(sy)})
    
    output_file = input_file.replace('.json','_smoothed.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_file

if __name__ == "__main__":
   start_time = time.time()
#    input_path = "leftBackhand_side_trajectory.json"
   input_path = "leftBackhand_45_trajectory.json"
   smoothed_data = smooth_2D_trajectory(input_path)
   print(f"Execution time: {time.time() - start_time:.4f}s")