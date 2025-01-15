import json
import numpy as np
from scipy.signal import savgol_filter
import time

def smooth_2D_trajectory(input_file, window_length=15, polyorder=3, angle_threshold=30):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Define all keypoints to smooth
    keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # Handle tennis ball trajectory first
    ball_frames = [(i, frame['tennis_ball']) for i, frame in enumerate(data) 
                   if frame['tennis_ball']['x'] is not None]
    
    if ball_frames:
        start_frame, end_frame = ball_frames[0][0], ball_frames[-1][0]
        segments = [(start_frame, end_frame)]
        
        # Interpolate missing ball positions
        for seg_start, seg_end in segments:
            coords = [(data[i]['tennis_ball']['x'], 
                      data[i]['tennis_ball']['y'], i)
                     for i in range(seg_start, seg_end + 1)
                     if data[i]['tennis_ball']['x'] is not None]
            
            if len(coords) > window_length:
                x, y, frames = zip(*coords)
                x_interp = np.interp(range(seg_start, seg_end + 1), frames, x)
                y_interp = np.interp(range(seg_start, seg_end + 1), frames, y)
                
                # Fill in interpolated values for missing frames
                for i, frame_idx in enumerate(range(seg_start, seg_end + 1)):
                    if data[frame_idx]['tennis_ball']['x'] is not None:
                        continue
                    data[frame_idx]['tennis_ball'].update({
                        'x': float(x_interp[i]),
                        'y': float(y_interp[i])
                    })
    
    # Smooth all body keypoints
    for keypoint in keypoints:
        # Extract coordinates
        x_coords = [frame[keypoint]['x'] for frame in data]
        y_coords = [frame[keypoint]['y'] for frame in data]
        
        # Check if we have enough valid points for smoothing
        valid_points = [i for i, (x, y) in enumerate(zip(x_coords, y_coords)) 
                       if x is not None and y is not None]
        
        if len(valid_points) > window_length:
            # Create masks for valid points
            x_array = np.array(x_coords)
            y_array = np.array(y_coords)
            
            # Replace None with neighboring values for smoothing
            for i in range(len(x_array)):
                if x_array[i] is None or y_array[i] is None:
                    # Find nearest valid point
                    valid_indices = np.where([x is not None and y is not None 
                                            for x, y in zip(x_array, y_array)])[0]
                    if len(valid_indices) > 0:
                        nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - i))]
                        x_array[i] = x_array[nearest_idx]
                        y_array[i] = y_array[nearest_idx]
            
            # Convert to float arrays
            x_array = x_array.astype(float)
            y_array = y_array.astype(float)
            
            # Apply Savitzky-Golay filter
            smooth_x = savgol_filter(x_array, window_length, polyorder)
            smooth_y = savgol_filter(y_array, window_length, polyorder)
            
            # Update the data with smoothed coordinates
            for i, (sx, sy) in enumerate(zip(smooth_x, smooth_y)):
                # Only update frames that originally had valid coordinates
                if x_coords[i] is not None and y_coords[i] is not None:
                    data[i][keypoint].update({
                        'x': float(sx),
                        'y': float(sy)
                    })
    
    # Save smoothed data
    output_file = input_file.replace('.json', '_smoothed.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_file

if __name__ == "__main__":
    start_time = time.time()
    # input_path = "leftBackhand_side_trajectory.json"
    input_path = "leftBackhand_45_trajectory.json"
    smoothed_data = smooth_2D_trajectory(input_path)
    print(f"Execution time: {time.time() - start_time:.4f}s")