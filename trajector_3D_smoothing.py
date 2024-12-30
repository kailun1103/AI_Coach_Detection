import json
import numpy as np
from scipy.signal import savgol_filter
import time

def smooth_3D_trajectory(input_path, window_length=15, polyorder=3, angle_threshold=30):
   with open(input_path, 'r') as f:
       data = json.load(f)

   # Smooth wrist trajectory
   coords = {axis: savgol_filter([frame['left_wrist'][axis] for frame in data], 
                                 window_length, polyorder)
            for axis in ['x', 'y', 'z']}

   # Get valid ball frames 
   ball_frames = [(i, frame['tennis_ball']) for i, frame in enumerate(data) 
                  if not any(frame['tennis_ball'][axis] is None for axis in ['x','y','z'])]
   
   if ball_frames:
       start_frame, end_frame = ball_frames[0][0], ball_frames[-1][0]
       
       # Get valid points
       valid_points = [[data[i]['tennis_ball']['x'], 
                       data[i]['tennis_ball']['y'],
                       data[i]['tennis_ball']['z'], i]
                      for i in range(start_frame, end_frame + 1)
                      if not any(data[i]['tennis_ball'][axis] is None for axis in ['x','y','z'])]
       
       # Find angle points
       angle_points = []
       for i in range(1, len(valid_points) - 1):
           p1, p2, p3 = map(lambda x: np.array(x[:3]), 
                           [valid_points[i-1], valid_points[i], valid_points[i+1]])
           v1, v2 = p1 - p2, p3 - p2
           angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / 
                            (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
           
           if angle < angle_threshold:
               angle_points.append(valid_points[i][3])
       
       # Create and smooth segments
       segments = [(last := start_frame, pt) for pt in sorted(angle_points)]
       segments.append((angle_points[-1] if angle_points else start_frame, end_frame))
       
       for seg_start, seg_end in segments:
           ball_coords = {axis: [] for axis in ['x','y','z']}
           frames = []
           
           for i in range(seg_start, seg_end + 1):
               ball = data[i]['tennis_ball']
               if not any(ball[axis] is None for axis in ['x','y','z']):
                   for axis in ['x','y','z']:
                       ball_coords[axis].append(ball[axis])
                   frames.append(i)
           
           if len(frames) > window_length:
               for axis in ['x','y','z']:
                   interp = np.interp(range(seg_start, seg_end + 1), frames, ball_coords[axis])
                   smoothed = savgol_filter(interp, window_length, polyorder)
                   
                   for i, frame_idx in enumerate(range(seg_start, seg_end + 1)):
                       data[frame_idx]['tennis_ball'][axis] = float(smoothed[i])
   
   # Update trajectory
   smoothed_data = [{
       'frame': data[i]['frame'],
       'left_wrist': {axis: float(coords[axis][i]) for axis in ['x','y','z']},
       'tennis_ball': data[i]['tennis_ball']
   } for i in range(len(data))]
   
   output_path = input_path.replace('.json','_smoothed.json')
   with open(output_path, 'w') as f:
       json.dump(smoothed_data, f, indent=2)
       
   return output_path

if __name__ == "__main__":
   start = time.time()
   input_path = "leftBackhand_3D_trajectory.json"
   output_path = smooth_3D_trajectory(input_path)
   print(f"Execution time: {time.time() - start:.4f}s")