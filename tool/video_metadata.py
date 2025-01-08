from pymediainfo import MediaInfo

def get_timecode_info(video_path):
    media_info = MediaInfo.parse(video_path)
    
    # 遍歷所有軌道尋找 Other Track 中的時間碼資訊
    for track in media_info.tracks:
        if track.track_type == "Other" and track.format == "QuickTime TC":
            # 取得幀數資訊
            frame_count = track.frame_count
            
            return {
                "first_frame": track.time_code_of_first_frame,
                "last_frame": track.time_code_of_last_frame,
                "total_frames": frame_count
            }
    
    return None

# 使用示例
video_path = "GX010240.MP4"
timecode_info = get_timecode_info(video_path)

if timecode_info:
    print(f"First Frame Timecode: {timecode_info['first_frame']}")
    print(f"Last Frame Timecode: {timecode_info['last_frame']}")
    print(f"Total Frames: {timecode_info['total_frames']}")
else:
    print("No timecode information found in the video file.")