import json
import numpy as np
from scipy.signal import savgol_filter
import time

def smooth_3D_trajectory(input_file, window_length=15, polyorder=3, tennis_window_length=7, tennis_polyorder=2):
    """
    Smooth 3D trajectory data for each keypoint using Savitzky-Golay filter
    
    Parameters:
    input_file (str): Path to input JSON file
    window_length (int): Window length for smoothing
    polyorder (int): Polynomial order for smoothing
    tennis_window_length (int): Window length for tennis ball smoothing
    tennis_polyorder (int): Polynomial order for tennis ball smoothing
    """
    # Read input data
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    # Get the original tennis_ball_hit and tennis_ball_angle data
    original_hit_data = [(frame['tennis_ball_hit'], frame['tennis_ball_angle']) 
                        for frame in data]
    
    # Define keypoints to smooth
    keypoints = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # First handle tennis ball trajectory
    first_valid = None
    last_valid = None
    for i, frame in enumerate(data):
        if all(frame['tennis_ball'][coord] is not None for coord in ['x', 'y', 'z']):
            if first_valid is None:
                first_valid = i
            last_valid = i

    if first_valid is not None and last_valid is not None:
        # Extract valid tennis ball data
        valid_range_data = data[first_valid:last_valid + 1]
        coords = {'x': [], 'y': [], 'z': []}
        
        # Get tennis ball coordinates
        for frame in valid_range_data:
            for coord in coords:
                coords[coord].append(frame['tennis_ball'][coord] if frame['tennis_ball'][coord] is not None else np.nan)
        
        # Convert to numpy arrays
        for coord in coords:
            coords[coord] = np.array(coords[coord], dtype=float)
        
        # Smooth each coordinate if enough valid points
        if len(coords['x']) > tennis_window_length:
            for coord in coords:
                coord_array = coords[coord]
                
                # Interpolate NaN values if any
                if np.any(np.isnan(coord_array)):
                    valid_indices = ~np.isnan(coord_array)
                    invalid_indices = np.isnan(coord_array)
                    
                    if np.any(valid_indices):
                        valid_positions = np.where(valid_indices)[0]
                        all_positions = np.arange(len(coord_array))
                        coord_array[invalid_indices] = np.interp(
                            all_positions[invalid_indices],
                            valid_positions,
                            coord_array[valid_indices]
                        )
                
                # Apply Savitzky-Golay filter
                smooth_coord = savgol_filter(coord_array, tennis_window_length, tennis_polyorder)
                
                # Update data
                for i, value in enumerate(smooth_coord):
                    data[i + first_valid]['tennis_ball'][coord] = float(value)

    # Process other keypoints
    for keypoint in keypoints:
        # Extract coordinates
        coords = {'x': [], 'y': [], 'z': []}
        for frame in data:
            point = frame[keypoint]
            coords['x'].append(point['x'] if point['x'] is not None else np.nan)
            coords['y'].append(point['y'] if point['y'] is not None else np.nan)
            coords['z'].append(point['z'] if point['z'] is not None else np.nan)
            
        # Convert to numpy arrays
        for coord in coords:
            coords[coord] = np.array(coords[coord], dtype=float)
            
        # Find valid points (not NaN)
        valid_points = ~np.isnan(coords['x'])
        if np.sum(valid_points) > window_length:
            # Handle each coordinate (x, y, z)
            for coord in coords:
                coord_array = coords[coord]
                
                # Interpolate NaN values
                if np.any(np.isnan(coord_array)):
                    valid_indices = ~np.isnan(coord_array)
                    invalid_indices = np.isnan(coord_array)
                    
                    if np.any(valid_indices):
                        valid_positions = np.where(valid_indices)[0]
                        all_positions = np.arange(len(coord_array))
                        coord_array[invalid_indices] = np.interp(
                            all_positions[invalid_indices],
                            valid_positions,
                            coord_array[valid_indices]
                        )
                
                # Apply Savitzky-Golay filter
                if len(coord_array) > window_length:
                    smooth_coord = savgol_filter(coord_array, window_length, polyorder)
                    
                    # Update data with smoothed values
                    for i, value in enumerate(smooth_coord):
                        if not np.isnan(coords[coord][i]):  # Only update if original value wasn't None
                            data[i][keypoint][coord] = float(value)
                            
    # Restore original tennis_ball_hit and tennis_ball_angle values
    for i, (hit, angle) in enumerate(original_hit_data):
        data[i]['tennis_ball_hit'] = hit
        data[i]['tennis_ball_angle'] = angle
    
    # Save smoothed data
    output_file = input_file.replace(').json', '_smoothed).json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_file

if __name__ == "__main__":
    start_time = time.time()
    
    # Specify input file path
    input_path = "temp/junior_3D_trajectory.json"  # Change this to your input file path
    
    # Process the trajectories
    output_path = smooth_3D_trajectory(input_path)
    
    print(f"Execution time: {time.time() - start_time:.4f}s")
    print(f"Smoothed data saved to: {output_path}")