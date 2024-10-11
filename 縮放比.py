import math

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# a 組數據
a_shoulder = (464, 240)
a_hip = (483, 404)

# b 組數據
b_shoulder = (729, 265)
b_hip = (746, 551)

# 計算兩組數據中肩部到臀部的距離
a_distance = calculate_distance(a_shoulder, a_hip)
b_distance = calculate_distance(b_shoulder, b_hip)

# 計算縮放比例
scale_factor = a_distance / b_distance

print(f"縮放比例: {scale_factor:.4f}")