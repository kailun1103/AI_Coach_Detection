from pymediainfo import MediaInfo
import time
import cv2
import os
from multiprocessing import Process, Queue

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

def process_video(video_path, start_frame, end_frame, output_path, progress_queue):
    """Process a single video"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = end_frame - start_frame + 1
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process frames
    processed_frames = 0
    for _ in range(total_frames):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            processed_frames += 1
            if processed_frames % 100 == 0:  # Update progress every 100 frames
                progress = (processed_frames / total_frames) * 100
                progress_queue.put((video_path, progress))
        else:
            break
    
    # Cleanup
    cap.release()
    out.release()
    progress_queue.put((video_path, 100))  # Signal completion

def save_synchronized_videos(video1_path, video2_path, start_frame1, end_frame1, start_frame2, end_frame2):
    """Save synchronized portions of videos using parallel processing"""
    # Create output directory if it doesn't exist
    output_dir = "synchronized_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get file names without extension
    video1_name = os.path.splitext(os.path.basename(video1_path))[0]
    video2_name = os.path.splitext(os.path.basename(video2_path))[0]
    
    # Define output paths
    output_path1 = os.path.join(output_dir, f"{video1_name}_sync.mp4")
    output_path2 = os.path.join(output_dir, f"{video2_name}_sync.mp4")
    
    # Create a queue for progress updates
    progress_queue = Queue()
    
    # Create and start processes
    p1 = Process(target=process_video, 
                 args=(video1_path, start_frame1, end_frame1, output_path1, progress_queue))
    p2 = Process(target=process_video, 
                 args=(video2_path, start_frame2, end_frame2, output_path2, progress_queue))
    
    p1.start()
    p2.start()
    
    # Track progress
    videos_completed = set()
    progress = {video1_path: 0, video2_path: 0}
    
    while len(videos_completed) < 2:
        video_path, percent = progress_queue.get()
        progress[video_path] = percent
        
        if percent == 100:
            videos_completed.add(video_path)
        
        # Print progress
        print(f"\rProgress - Video 1: {progress[video1_path]:.1f}%, " 
              f"Video 2: {progress[video2_path]:.1f}%", end="")
    
    print("\n")  # New line after progress
    
    # Wait for processes to complete
    p1.join()
    p2.join()
    
    print(f"\nSynchronized videos saved in '{output_dir}' directory")
    print(f"Video 1: {video1_name}_sync.mp4")
    print(f"Video 2: {video2_name}_sync.mp4")

if __name__ == "__main__":
    # Set video paths
    video1_path = "0315_45.MP4"
    video2_path = "0315_side.MP4"
    
    start_time = time.time()  # Start timer
    
    # Execute analysis
    result = analyze_timecode(video1_path, video2_path)
    if result:
        start_frame1, end_frame1, start_frame2, end_frame2 = result
        
        print(f"Return values:\n")
        print(video1_path)
        print(f"start_frame1: {start_frame1}")
        print(f"end_frame1: {end_frame1}")
        print('------------------------------')
        print(video2_path)
        print(f"start_frame2: {start_frame2}")
        print(f"end_frame2: {end_frame2}")
        
        # Save synchronized videos
        print("\nSaving synchronized videos...")
        save_synchronized_videos(video1_path, video2_path, 
                               start_frame1, end_frame1, 
                               start_frame2, end_frame2)
    
    end_time = time.time()  # End timer
    execution_time = end_time - start_time  # Calculate execution time
    print(f"\nExecution time: {execution_time:.4f} seconds")