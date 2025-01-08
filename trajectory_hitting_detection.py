import json
import math

def calculate_angle(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    vector1 = (x2 - x1, y2 - y1, z2 - z1)
    vector2 = (x3 - x2, y3 - y2, z3 - z2)
    
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(v**2 for v in vector1))
    magnitude2 = math.sqrt(sum(v**2 for v in vector2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    cos_theta = dot_product / (magnitude1 * magnitude2)
    return math.degrees(math.acos(max(min(cos_theta, 1), -1)))

def find_most_likely_hit_frame(data):
    max_angle = 0
    hit_frame = None
    
    valid_frames = [
        frame for frame in data
        if frame["tennis_ball"]["x"] is not None
        and frame["tennis_ball"]["y"] is not None
        and frame["tennis_ball"]["z"] is not None
    ]
    
    for i in range(1, len(valid_frames) - 1):
        x1, y1, z1 = valid_frames[i - 1]["tennis_ball"].values()
        x2, y2, z2 = valid_frames[i]["tennis_ball"].values()
        x3, y3, z3 = valid_frames[i + 1]["tennis_ball"].values()
        
        angle = calculate_angle(x1, y1, z1, x2, y2, z2, x3, y3, z3)
        
        if angle > max_angle:
            max_angle = angle
            hit_frame = valid_frames[i]["frame"]
    
    return hit_frame

def add_tennis_hit_flag(input_file):
    # 讀取原始 JSON 檔案
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # 找到最可能的擊球幀
    hit_frame = find_most_likely_hit_frame(data)
    
    # 為每一幀添加 tennis_hit 標記
    for frame in data:
        frame["tennis_hit"] = (frame["frame"] == hit_frame)
    
    # 直接寫回原始檔案
    with open(input_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        


if __name__ == "__main__":
    input_file = 'leftBackhand_3D_trajectory.json'
    add_tennis_hit_flag(input_file)