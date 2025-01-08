import json
import numpy as np

def interpolate_trajectory(json_file):
    """
    Analyze trajectory data, interpolate missing values, and save back to the original file
    
    Parameters:
    json_file (str): Input JSON file path
    """
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert data to numpy arrays for interpolation
    frames = np.array([frame['frame'] for frame in data])
    
    # Process left wrist data
    lw_x = np.array([frame['left_wrist']['x'] for frame in data], dtype=float)
    lw_y = np.array([frame['left_wrist']['y'] for frame in data], dtype=float)
    
    # Process tennis ball data
    tb_x = np.array([frame['tennis_ball']['x'] for frame in data], dtype=float)
    tb_y = np.array([frame['tennis_ball']['y'] for frame in data], dtype=float)
    
    # Interpolate left wrist data
    valid_lw = ~np.isnan(lw_x)  # Find indices of non-null values
    if np.any(~valid_lw):  # If there are missing values
        lw_x[~valid_lw] = np.interp(frames[~valid_lw], frames[valid_lw], lw_x[valid_lw])
        lw_y[~valid_lw] = np.interp(frames[~valid_lw], frames[valid_lw], lw_y[valid_lw])
    
    # Interpolate tennis ball data
    valid_tb = ~np.isnan(tb_x)  # Find indices of non-null values
    if np.any(~valid_tb):  # If there are missing values
        tb_x[~valid_tb] = np.interp(frames[~valid_tb], frames[valid_tb], tb_x[valid_tb])
        tb_y[~valid_tb] = np.interp(frames[~valid_tb], frames[valid_tb], tb_y[valid_tb])
    
    # Update original data
    for i, frame in enumerate(data):
        # Update left wrist data
        if frame['left_wrist']['x'] is None or frame['left_wrist']['y'] is None:
            frame['left_wrist']['x'] = float(lw_x[i])
            frame['left_wrist']['y'] = float(lw_y[i])
        
        # Update tennis ball data
        if frame['tennis_ball']['x'] is None or frame['tennis_ball']['y'] is None:
            frame['tennis_ball']['x'] = float(tb_x[i])
            frame['tennis_ball']['y'] = float(tb_y[i])

    # Save interpolated data back to the original file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    input_file = "leftBackhand_45_trajectory.json"
    interpolate_trajectory(input_file)