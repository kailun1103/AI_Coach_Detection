import json
import numpy as np

def load_json_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def find_hit_frame(data_3d):
    for frame_data in data_3d:
        if frame_data['tennis_hit']:
            return frame_data['frame']
    return None

def correct_3d_coordinates(data_3d, data_2d):
    # Find the hit frame
    hit_frame = find_hit_frame(data_3d)
    if hit_frame is None:
        raise ValueError("No tennis hit frame found in 3D data")
    
    # Create coordinate mapping between 2D and 3D frames
    frame_2d_mapping = {frame_data['frame']: frame_data for frame_data in data_2d}
    
    # Calculate scale factors using hit frame as reference
    hit_frame_2d = frame_2d_mapping[hit_frame]
    hit_frame_3d = next(f for f in data_3d if f['frame'] == hit_frame)
    
    # Scale factors for y and z coordinates
    scale_y = abs(hit_frame_3d['tennis_ball']['y'] / hit_frame_2d['tennis_ball']['x'])
    scale_z = abs(hit_frame_3d['tennis_ball']['z'] / hit_frame_2d['tennis_ball']['y'])
    
    # Get the first and last frame's original y value
    first_frame_3d = data_3d[0]
    first_frame_y = first_frame_3d['tennis_ball']['y']
    last_frame_3d = data_3d[-1]
    last_frame_y = last_frame_3d['tennis_ball']['y']
    
    # Find hit frame index and total frames before and after hit
    hit_frame_index = next(i for i, f in enumerate(data_3d) if f['tennis_hit'])
    total_frames_before_hit = hit_frame_index
    total_frames_after_hit = len(data_3d) - hit_frame_index
    
    # Correct coordinates for each frame
    corrected_data = []
    hit_occurred = False
    frames_before_hit = 0
    frames_since_hit = 0
    
    # Calculate the hit y position
    hit_y = hit_frame_2d['tennis_ball']['x'] * scale_y
    # Calculate midpoint y value with offset
    offset_factor = -10  # 調整這個值可以控制向右偏移的程度
    mid_y = (first_frame_y + hit_y) / 2 + offset_factor
    
    for frame_3d in data_3d:
        frame_num = frame_3d['frame']
        if frame_num in frame_2d_mapping:
            frame_2d = frame_2d_mapping[frame_num]
            
            if frame_3d['tennis_hit']:
                hit_occurred = True
                hit_y = frame_2d['tennis_ball']['x'] * scale_y
            
            # Calculate y coordinate based on whether hit has occurred
            if not hit_occurred:
                # Before hit: Create a curved path using quadratic interpolation
                progress = frames_before_hit / total_frames_before_hit
                
                # Quadratic interpolation for smoother curve
                if progress <= 0.5:
                    # First half: interpolate between start and mid point
                    t = progress * 2
                    y_coord = first_frame_y * (1 - t) + mid_y * t
                else:
                    # Second half: interpolate between mid point and hit point
                    t = (progress - 0.5) * 2
                    y_coord = mid_y * (1 - t) + hit_y * t
                
                frames_before_hit += 1
            else:
                # After hit: Same as before, interpolate between hit position and final position
                progress = frames_since_hit / total_frames_after_hit
                start_y = hit_y
                y_coord = start_y + (last_frame_y - start_y) * progress
                frames_since_hit += 1
            
            # Create new frame data with corrected coordinates
            new_frame = {
                'frame': frame_num,
                'left_wrist': frame_3d['left_wrist'],
                'tennis_ball': {
                    'x': frame_3d['tennis_ball']['x'],
                    'y': y_coord,
                    'z': -frame_2d['tennis_ball']['y'] * scale_z
                },
                'tennis_hit': frame_3d['tennis_hit']
            }
            corrected_data.append(new_frame)
    
    return corrected_data

def main():
    # Load the data
    data_3d = load_json_file('leftBackhand_3D_trajectory_smoothed.json')
    data_2d = load_json_file('leftBackhand_side_trajectory_smoothed.json')
    
    # Correct the coordinates
    corrected_data = correct_3d_coordinates(data_3d, data_2d)
    
    # Save the corrected data
    save_json_file(corrected_data, 'leftBackhand_3D_trajectory_corrected.json')
    print("Correction completed. Data saved to 'leftBackhand_3D_trajectory_corrected.json'")

if __name__ == "__main__":
    main()