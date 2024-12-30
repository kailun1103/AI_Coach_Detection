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

def process_trajectories(left_path, leftfront_path, P1, P2):
   # Load trajectory data
   with open(left_path) as f1, open(leftfront_path) as f2:
       left_data = json.load(f1)
       leftfront_data = json.load(f2)

   points_3d = []
  
   for frame_idx, (left_point, leftfront_point) in enumerate(zip(left_data, leftfront_data)):
       # Calculate wrist 3D coordinates
       wrist1 = np.array([left_point['left_wrist']['x'], left_point['left_wrist']['y']])
       wrist2 = np.array([leftfront_point['left_wrist']['x'], leftfront_point['left_wrist']['y']])
       wrist_3d = triangulate_point(P1, P2, wrist1, wrist2)
       
       # Calculate ball 3D coordinates if valid
       ball_3d = None
       if all(p['tennis_ball']['x'] is not None for p in (left_point, leftfront_point)):
           ball1 = np.array([left_point['tennis_ball']['x'], left_point['tennis_ball']['y']])
           ball2 = np.array([leftfront_point['tennis_ball']['x'], leftfront_point['tennis_ball']['y']])
           ball_3d = triangulate_point(P1, P2, ball1, ball2)
       
       # Format frame data
       points_3d.append({
           'frame': frame_idx,
           'left_wrist': dict(zip(['x','y','z'], map(float, [wrist_3d[0], -wrist_3d[1], -wrist_3d[2]]))),
           'tennis_ball': dict(zip(['x','y','z'], map(float, [ball_3d[0], -ball_3d[1], -ball_3d[2]]))) if ball_3d is not None 
           else {'x': None, 'y': None, 'z': None}
       })
  
   output_path = 'leftBackhand_3D_trajectory.json'
   with open(output_path, 'w') as f:
       json.dump(points_3d, f, indent=2)
   
   return output_path

if __name__ == "__main__":
    start = time.perf_counter()
    
    # Define projection matrices in main
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

    input_path_1 = 'leftBackhand_side_trajectory.json'
    input_path_2 = 'leftBackhand_45_trajectory.json'
    output_path = process_trajectories(input_path_1, input_path_2, P1, P2)
    
    print(f"Execution time: {time.perf_counter() - start:.4f}s")