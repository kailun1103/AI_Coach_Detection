import json

def find_hit_frame(data):
    """Find the frame index where tennis_ball_hit is True"""
    for i, frame in enumerate(data):
        if frame.get('tennis_ball_hit', False):
            return i
    return -1

def sync_trajectories(file1, file2):
    """
    Synchronize two trajectories based on tennis_ball_hit frame and reindex frames
    Args:
        file1: First trajectory file path
        file2: Second trajectory file path
    """
    # Read the input files
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    # Find hit frames
    hit_frame1 = find_hit_frame(data1)
    hit_frame2 = find_hit_frame(data2)
    
    if hit_frame1 == -1 or hit_frame2 == -1:
        raise ValueError("Could not find tennis_ball_hit in one or both trajectories")
    
    # Calculate frames to keep before hit
    frames_before = min(hit_frame1, hit_frame2)
    
    # Calculate frames to keep after hit
    frames_after = min(len(data1) - hit_frame1 - 1, len(data2) - hit_frame2 - 1)
    
    # Extract synchronized segments
    start1 = hit_frame1 - frames_before
    end1 = hit_frame1 + frames_after + 1
    start2 = hit_frame2 - frames_before
    end2 = hit_frame2 + frames_after + 1
    
    synced_data1 = data1[start1:end1]
    synced_data2 = data2[start2:end2]
    
    # Reindex frames starting from 0
    for i, frame in enumerate(synced_data1):
        frame['frame'] = i
    
    for i, frame in enumerate(synced_data2):
        frame['frame'] = i
    
    # Write synchronized data back to the original files
    with open(file1, 'w') as f:
        json.dump(synced_data1, f, indent=2)
    with open(file2, 'w') as f:
        json.dump(synced_data2, f, indent=2)

    
if __name__ == "__main__":
    file1 = 'temp/junior_side_trajectory_smoothed.json'
    file2 = 'temp/junior_45_trajectory_smoothed.json'
    sync_trajectories(file1, file2)