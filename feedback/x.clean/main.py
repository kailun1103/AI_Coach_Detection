import os
import pandas as pd
import json

class JsonClean:
    def __init__(self, b):
        self.b = b  
        
        self.junior_file_name = f"junior_9_{self.b}(3D_trajectory_smoothed)"
        self.junior_file_path = f"./__data__/raw/{self.junior_file_name}.json"
        
        self.pro_file_name = f"pro_test"
        self.pro_file_path = f"./__data__/pro/{self.pro_file_name}.json"
        
        self.save_path = f"./__data__/junior/junior_9_{self.b}_cleaned.json"

    def frame_extract(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
        
        # Extract specific fields
        frames = []
        for frame in data:
            extracted = {
                "frame": frame.get("frame"),
                "left_eye": frame.get("left_eye"),  
                "right_eye": frame.get("right_eye"),
                
                "right_wrist": frame.get("right_wrist"),
                "right_elbow": frame.get("right_elbow"),
                "right_shoulder": frame.get("right_shoulder"),
                
                "left_knee": frame.get("left_knee"),
                "right_knee": frame.get("right_knee"),
                
                "tennis_ball_hit": frame.get("tennis_ball_hit"),  # 加上逗號

            }
            frames.append(extracted)
        
        return frames

    def frame_clean(self, frames):
        # Round coordinates to 2 decimal places
        cleaned_frames = []
        for frame in frames:
            cleaned_frame = {
                key: {
                    k: round(v, 2) if isinstance(v, (int, float)) else v
                    for k, v in value.items()
                } if isinstance(value, dict) else value
                for key, value in frame.items()
            }
            cleaned_frames.append(cleaned_frame)
        
        return cleaned_frames

    def save(self, frames):
        # Save the cleaned frames to a JSON file
        with open(self.save_path, 'w') as f:
            json.dump(frames, f, indent=4)

    def process_file(self, file):
        extracted_frames = self.frame_extract(file)
        cleaned_frames = self.frame_clean(extracted_frames)
        self.save(cleaned_frames)
      
    def self_compare(self, frames):
        diff_frames = []
        
        for i in range(1, len(frames) - 1):
            current_frame = frames[i]
            previous_frame = frames[i - 1]
            diff_frame = {
                key: {
                    k: round(current_frame[key][k] - previous_frame[key][k], 2) 
                    if (key in current_frame and key in previous_frame and 
                        isinstance(current_frame[key], dict) and 
                        k in current_frame[key] and k in previous_frame[key] and 
                        isinstance(current_frame[key][k], (int, float)) and 
                        isinstance(previous_frame[key][k], (int, float)))
                    else "null"
                    for k in current_frame[key]
                } if isinstance(current_frame[key], dict) else current_frame[key]
                for key in current_frame
            }
            
            diff_frames.append(diff_frame)

        return diff_frames

    def main(self):
        if os.path.exists(self.junior_file_path):
            print(f"Processing file: {self.junior_file_path}")
            frames = self.frame_extract(self.junior_file_path)
            frames = self.frame_clean(frames)
            frames = self.self_compare(frames)
            self.save(frames)
        else:
            print(f"File not found: {self.junior_file_path}")
          

if __name__ == "__main__":
    for i in range(10):
        instance = JsonClean(i)
        instance.main()
