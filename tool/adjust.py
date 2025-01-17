import json
import numpy as np

# 讀取JSON文件
with open('leftBackhand_3D_trajectory_smoothed.json', 'r') as file:
    data = json.load(file)

# 尋找tennis_hit為true的frame
hit_frame = 0
for i, frame_data in enumerate(data):
    if frame_data.get('tennis_hit', False):
        hit_frame = i
        print(f"找到擊球時刻: frame {hit_frame}")
        break

# Part 1: 處理frame 0到hit_frame的拋物線
start_x = data[0]['tennis_ball']['x']
start_y = data[0]['tennis_ball']['y']  # 添加y座標
start_z = data[0]['tennis_ball']['z']
end_x = data[hit_frame]['tennis_ball']['x']
end_y = data[hit_frame]['tennis_ball']['y']  # 添加y座標
end_z = data[hit_frame]['tennis_ball']['z']

# 為frame 1創建平滑的過渡點
transition_ratio = 0.0005
data[1]['tennis_ball']['x'] = start_x + (end_x - start_x) * transition_ratio
data[1]['tennis_ball']['y'] = start_y + (end_y - start_y) * transition_ratio
data[1]['tennis_ball']['z'] = start_z + (end_z - start_z) * transition_ratio

# 修改拋物線生成方式
frames_to_modify = range(2, hit_frame)
total_frames = len(frames_to_modify)

for i, frame in enumerate(frames_to_modify):
    t = i / (total_frames - 1)
    
    # 平滑插值函數
    t = t * t * (3 - 2 * t)
    
    # 線性插值
    original_x = start_x + (end_x - start_x) * t
    original_y = start_y + (end_y - start_y) * t
    original_z = start_z + (end_z - start_z) * t
    
    # 修改偏移計算，降低y方向的高度
    x_offset = 180 * np.sin(t * np.pi)
    y_offset = 800 * np.sin(t * np.pi)  # 降低y方向的最大偏移值，原來是沒有的
    z_offset = 200 * np.sin(t * np.pi)
    
    # 更新座標
    data[frame]['tennis_ball']['x'] = original_x + x_offset
    data[frame]['tennis_ball']['y'] = original_y + y_offset  # 添加y偏移
    data[frame]['tennis_ball']['z'] = original_z - z_offset

# Part 2: 處理hit_frame到最後的直線
last_frame = len(data) - 1
hit_frame_x = data[hit_frame]['tennis_ball']['x']
hit_frame_y = data[hit_frame]['tennis_ball']['y']
hit_frame_z = data[hit_frame]['tennis_ball']['z']

final_y = data[last_frame]['tennis_ball']['y'] + 300
last_frame_x = data[last_frame]['tennis_ball']['x']
last_frame_z = data[last_frame]['tennis_ball']['z']

for frame in range(hit_frame + 1, last_frame + 1):
    t = (frame - hit_frame) / (last_frame - hit_frame)
    
    new_x = hit_frame_x + (last_frame_x - hit_frame_x) * t
    new_y = hit_frame_y + (final_y - hit_frame_y) * t
    new_z = hit_frame_z + (last_frame_z - hit_frame_z) * t
    
    data[frame]['tennis_ball']['x'] = new_x
    data[frame]['tennis_ball']['y'] = new_y
    data[frame]['tennis_ball']['z'] = new_z

# 保存修改後的JSON文件
with open('leftBackhand_3D_trajectory_adjusted.json', 'w') as file:
    json.dump(data, file, indent=2)

print(f"軌跡已成功修改！以 frame {hit_frame} 為分界點")
print(f"最後一幀的 y 座標已調整為: {final_y}")