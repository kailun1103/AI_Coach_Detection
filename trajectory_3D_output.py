import numpy as np
import json
import time

def triangulate_point(P1, P2, point1, point2):
    A = np.zeros((4, 4))
    A[0] = point1[1] * P1[2] - P1[1]
    A[1] = P1[0] - point1[0] * P1[2]
    A[2] = point2[1] * P2[2] - P2[1]
    A[3] = P2[0] - point2[0] * P2[2]
    
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def fix_trajectory(data):
    start_frame = None
    end_frame = None
    
    for i, frame in enumerate(data):
        if frame['tennis_ball_hit']:
            start_frame = i
        if frame['tennis_ball']['x'] is not None:
            end_frame = i
            
    if start_frame is None or end_frame is None:
        return data
        
    start_pos = data[start_frame]['tennis_ball']
    end_pos = data[end_frame]['tennis_ball']
    num_frames = end_frame - start_frame + 1
    
    for axis in ['x', 'y', 'z']:
        start_val = start_pos[axis]
        end_val = end_pos[axis]
        values = np.linspace(start_val, end_val, num_frames)
        
        for i, val in enumerate(values):
            frame_idx = start_frame + i
            data[frame_idx]['tennis_ball'][axis] = float(val)
            
    return data

def process_trajectories(left_path, leftfront_path, P1, P2):
    keypoints = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
        'right_knee', 'left_ankle', 'right_ankle', 'tennis_ball'
    ]

    with open(left_path) as f1, open(leftfront_path) as f2:
        left_data = json.load(f1)
        leftfront_data = json.load(f2)

    points_3d = []
    
    for frame_idx, (left_point, leftfront_point) in enumerate(zip(left_data, leftfront_data)):
        frame_data = {'frame': frame_idx}
        
        for keypoint in keypoints:
            point_3d = None
            
            if (left_point[keypoint]['x'] is not None and 
                left_point[keypoint]['y'] is not None and
                leftfront_point[keypoint]['x'] is not None and 
                leftfront_point[keypoint]['y'] is not None):
                
                point1 = np.array([left_point[keypoint]['x'], left_point[keypoint]['y']])
                point2 = np.array([leftfront_point[keypoint]['x'], leftfront_point[keypoint]['y']])
                
                try:
                    point_3d = triangulate_point(P1, P2, point1, point2)
                    point_3d = {
                        'x': float(point_3d[0]),
                        'y': float(-point_3d[1]),
                        'z': float(point_3d[2])
                    }
                except:
                    point_3d = {'x': None, 'y': None, 'z': None}
            else:
                point_3d = {'x': None, 'y': None, 'z': None}
            
            frame_data[keypoint] = point_3d
        
        frame_data['tennis_ball_hit'] = left_point['tennis_ball_hit']
        frame_data['tennis_ball_angle'] = left_point['tennis_ball_angle']
        
        points_3d.append(frame_data)

    # 首先執行3D重建
    parts = leftfront_path.split('_45')
    second_part = parts[1].split('(')[0] 
    output_path = parts[0] + second_part + '(3D_trajectory).json'
    
    # 然後修復軌跡
    fixed_data = fix_trajectory(points_3d)
    
    with open(output_path, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    return output_path

if __name__ == "__main__":
    start = time.perf_counter()
    
    P1 = np.array([
        [5830.127771, 0, 2707.891358, 0],
        [0, 5660.852212, 2650.794043, 0],
        [0, 0, 1, 0] 
    ])
   
    P2 = np.array([
        [-127.726676, -549.005678, 4533.763086, -23449322.445458],
        [-1883.494533, 3034.703903, 1416.289936, 6432610.718249],
        [-0.860417, -0.091385, 0.501330, 2218.320368]
    ])

    input_path_1 = 'temp/junior_side_trajectory_smoothed.json'
    input_path_2 = 'temp/junior_45_trajectory_smoothed.json'
    
    output_path = process_trajectories(input_path_1, input_path_2, P1, P2)
    print(f"Execution time: {time.perf_counter() - start:.4f}s")