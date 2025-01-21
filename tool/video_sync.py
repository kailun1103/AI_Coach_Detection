from pymediainfo import MediaInfo
import time

def get_relative_frame_number(start_tc, target_tc, fps=59.94):
    """Calculate relative frame number"""
    def tc_to_frames(tc):
        hours, minutes, seconds, frames = map(int, tc.split(':'))
        return hours * 3600 * fps + minutes * 60 * fps + seconds * fps + frames

    start_frames = tc_to_frames(start_tc)
    target_frames = tc_to_frames(target_tc)
    return int(target_frames - start_frames)

def analyze_timecode(video1_path, video2_path):
    # Analyze video timecode
    info1 = MediaInfo.parse(video1_path)
    info2 = MediaInfo.parse(video2_path)
    
    tc1 = None
    tc2 = None
    
    # Get timecode for first video
    for track in info1.tracks:
        if track.track_type == "Other" and track.format == "QuickTime TC":
            tc1 = {
                "start": track.time_code_of_first_frame,
                "end": track.time_code_of_last_frame
            }
    
    # Get timecode for second video
    for track in info2.tracks:
        if track.track_type == "Other" and track.format == "QuickTime TC":
            tc2 = {
                "start": track.time_code_of_first_frame,
                "end": track.time_code_of_last_frame
            }
    
    if not tc1 or not tc2:
        print("Error: Unable to get timecode information")
        return None
    
    # Calculate sync points
    sync_start = max(tc1['start'], tc2['start'])
    sync_end = min(tc1['end'], tc2['end'])
    
    # Calculate relative frame numbers for each video
    start_frame1 = get_relative_frame_number(tc1['start'], sync_start)
    end_frame1 = get_relative_frame_number(tc1['start'], sync_end)
    start_frame2 = get_relative_frame_number(tc2['start'], sync_start)
    end_frame2 = get_relative_frame_number(tc2['start'], sync_end)
    
    return start_frame1, end_frame1, start_frame2, end_frame2

if __name__ == "__main__":
    # Set video paths
    video1_path = "E:/運科影片/0109_Dataset/正拍/45正拍/1.mp4"
    video2_path = "E:\運科影片/0109_Dataset/正拍/側面正拍/1.mp4"
    
    start_time = time.time()  # Start timer
    
    # Execute analysis
    start_frame1, end_frame1, start_frame2, end_frame2 = analyze_timecode(video1_path, video2_path)
    
    print(f"Return values:\n")
    print(video1_path)
    print(f"start_frame1: {start_frame1}")
    print(f"end_frame1: {end_frame1}")
    print('------------------------------')
    print(video2_path)
    print(f"start_frame2: {start_frame2}")
    print(f"end_frame2: {end_frame2}")
    
    end_time = time.time()  # End timer
    execution_time = end_time - start_time  # Calculate execution time
    print(f"\nExecution time: {execution_time:.4f} seconds")