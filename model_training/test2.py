import numpy as np 
import json
import time

def create_projection_matrices():
    P1 = np.array(
        [[2153.578819,    0.000000, 2642.243210,    0.000000],
        [   0.000000, 2174.968926, 2665.481050,    0.000000],
        [   0.000000,    0.000000,    1.000000,    0.000000]]
    )
   
    P2 = np.array([
        [     515.377667,     -853.344207,     3473.079193, -1471578.582995],
        [   -1234.505665,     1603.723006,     2215.138835,  1946960.098824],
        [      -0.497978,       -0.270504,        0.823921,      738.808193]])
   
    return P1, P2

def triangulate_point(P1, P2, point1, point2):
    A = np.zeros((4, 4))
    A[0] = point1[1] * P1[2] - P1[1]
    A[1] = P1[0] - point1[0] * P1[2]
    A[2] = point2[1] * P2[2] - P2[1]
    A[3] = P2[0] - point2[0] * P2[2]
   
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def process_trajectories(left_path, leftfront_path):
    # Load trajectory data
    with open(left_path) as f1, open(leftfront_path) as f2:
        left_data = json.load(f1)
        leftfront_data = json.load(f2)

    P1, P2 = create_projection_matrices()
    points_3d = []
  
    for frame_idx, (left_point, leftfront_point) in enumerate(zip(left_data, leftfront_data)):
        # Calculate ball 3D coordinates if valid
        ball_3d = None
        if all(p['tennis_ball']['x'] is not None for p in (left_point, leftfront_point)):
            ball1 = np.array([left_point['tennis_ball']['x'], left_point['tennis_ball']['y']])
            ball2 = np.array([leftfront_point['tennis_ball']['x'], leftfront_point['tennis_ball']['y']])
            ball_3d = triangulate_point(P1, P2, ball1, ball2)
       
        # Format frame data
        points_3d.append({
            'frame': frame_idx,
            'tennis_ball': dict(zip(['x','y','z'], map(float, [ball_3d[0], -ball_3d[1], -ball_3d[2]]))) if ball_3d is not None 
            else {'x': None, 'y': None, 'z': None}
        })
  
    output_path = '3D_ball_trajectory.json'
    with open(output_path, 'w') as f:
        json.dump(points_3d, f, indent=2)
   
    return output_path

if __name__ == "__main__":
    start = time.perf_counter()
    input_path_1 = 'left_trajectory.json'
    input_path_2 = 'leftFront_trajectory.json'
    output_path = process_trajectories(input_path_1, input_path_2)
    print(f"Execution time: {time.perf_counter() - start:.4f}s")