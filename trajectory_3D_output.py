import numpy as np
import json
import time

def triangulate_point(P1, P2, point1, point2):
    """
    Triangulate a 3D point from two 2D points and projection matrices
    """
    A = np.zeros((4, 4))
    A[0] = point1[1] * P1[2] - P1[1]
    A[1] = P1[0] - point1[0] * P1[2]
    A[2] = point2[1] * P2[2] - P2[1]
    A[3] = P2[0] - point2[0] * P2[2]
    
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def process_trajectories(left_path, leftfront_path, P1, P2):
    # List of all keypoints to process
    keypoints = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
        'right_knee', 'left_ankle', 'right_ankle', 'tennis_ball'
    ]

    # Load trajectory data
    with open(left_path) as f1, open(leftfront_path) as f2:
        left_data = json.load(f1)
        leftfront_data = json.load(f2)

    points_3d = []
    
    for frame_idx, (left_point, leftfront_point) in enumerate(zip(left_data, leftfront_data)):
        frame_data = {'frame': frame_idx}
        
        # Process each keypoint
        for keypoint in keypoints:
            point_3d = None
            
            # Check if both views have valid coordinates for this keypoint
            if (left_point[keypoint]['x'] is not None and 
                left_point[keypoint]['y'] is not None and
                leftfront_point[keypoint]['x'] is not None and 
                leftfront_point[keypoint]['y'] is not None):
                
                point1 = np.array([left_point[keypoint]['x'], left_point[keypoint]['y']])
                point2 = np.array([leftfront_point[keypoint]['x'], leftfront_point[keypoint]['y']])
                
                try:
                    point_3d = triangulate_point(P1, P2, point1, point2)
                    # Convert to float and flip y and z coordinates as in original code
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
        
        # Copy tennis_ball_hit and tennis_ball_angle from left_data
        frame_data['tennis_ball_hit'] = left_point['tennis_ball_hit']
        frame_data['tennis_ball_angle'] = left_point['tennis_ball_angle']
        
        points_3d.append(frame_data)

    # Create output path by modifying input path
    leftfront_path = 'junior_9/forehand/9_0/junior_45_9_0(2D_trajectory_smoothed).json'
    # 先分割掉 "45" 這部分
    parts = leftfront_path.split('_45')
    # 從第二部分取出 "_9_0"
    second_part = parts[1].split('(')[0]  # 這會得到 "_9_0"
    # 組合最終路徑
    output_path = parts[0] + second_part + '(3D_trajectory).json'

    # output_path = leftfront_path.replace('(2D_trajectory_smoothed).json', '(3D_trajectory).json')
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(points_3d, f, indent=2)
    
    return output_path

if __name__ == "__main__":
    start = time.perf_counter()
    
    # Define projection matrices
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

    # File paths
    input_path_1 = 'temp/junior_side_trajectory_smoothed.json'
    input_path_2 = 'temp/junior_45_trajectory_smoothed.json'
    
    # Process the trajectories
    output_path = process_trajectories(input_path_1, input_path_2, P1, P2)
    
    print(f"Execution time: {time.perf_counter() - start:.4f}s")