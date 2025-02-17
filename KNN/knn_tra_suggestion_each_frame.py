# 運科KNN 

# 建立資料集：
# 標記資料
# 資料集結合

# 載入json file 的模型
# 1.將揮拍軌跡加入模型
# 2.一同做正規化
# 3.計算頭、左右肩膀、右手肘、右手腕 、左右膝蓋x y z的距離 （歐式距離）
# 4.選出最靠近的那個點
# 5.return最靠近點的建議

import json 
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 讀取 JSON 檔案
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# 寫入 JSON 檔案
def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# 標準化座標數據（merged_dataset + trajectory_data）
def normalize_data(dataset, trajectory_filename):
    keys = ["nose", "left_shoulder", "right_shoulder", "right_elbow", "right_wrist", "left_knee", "right_knee"]
    
    # 取得所有 frame 的數據
    data_matrix = {key: [] for key in keys}
    all_frames = []
    trajectory_frames = None  # 儲存 trajectory_data 內的 frames

    # 收集所有 frame
    for data in dataset:
        if data["filename"] == trajectory_filename:  
            trajectory_frames = data["data"]  # 記錄 trajectory_data 的 frames
        
        for frame in data["data"]:
            all_frames.append(frame)
            for key in keys:
                if key in frame and all(k in frame[key] for k in ["x", "y", "z"]):  # 確保 key 存在
                    data_matrix[key].append([frame[key]["x"], frame[key]["y"], frame[key]["z"]])

    # 初始化標準化器並對每個 key 進行標準化
    scalers = {key: MinMaxScaler() for key in keys}
    normalized_data = {key: scalers[key].fit_transform(data_matrix[key]) for key in keys}

    # 更新原始 JSON 數據
    frame_idx = 0
    for data in dataset:
        for i, frame in enumerate(data["data"]):
            for key in keys:
                if key in frame:
                    frame[key]["x"], frame[key]["y"], frame[key]["z"] = normalized_data[key][frame_idx]
            frame_idx += 1

    return trajectory_frames  # 回傳標準化後的 trajectory_data

# KNN 計算最短距離，回傳最接近的 filename 的 suggestion
def find_nearest_suggestion(trajectory_frames, merged_dataset, trajectory_filename):
    keys = ["nose", "left_shoulder", "right_shoulder", "right_elbow", "right_wrist", "left_knee", "right_knee"]
    filename_distances = {}

    for data in merged_dataset:
        if data["filename"] == trajectory_filename:
            continue  # 忽略自己，避免比對自身

        total_distance = 0  # 累積所有 frame 的歐式距離
        valid_comparisons = 0  # 記錄有效的 frame 比對數

        for traj_frame in trajectory_frames:
            min_frame_distance = float("inf")  # 紀錄 trajectory_frame 對應的最小 frame 距離

            for ref_frame in data["data"]:
                distance = 0
                for key in keys:
                    if key in traj_frame and key in ref_frame:
                        traj_point = traj_frame[key]
                        ref_point = ref_frame[key]

                        # 檢查是否有 None 值，若有則跳過
                        if any(v is None for v in traj_point.values()) or any(v is None for v in ref_point.values()):
                            continue

                        traj_array = np.array([traj_point["x"], traj_point["y"], traj_point["z"]])
                        ref_array = np.array([ref_point["x"], ref_point["y"], ref_point["z"]])
                        distance += np.linalg.norm(traj_array - ref_array)  # 歐式距離

                if distance < min_frame_distance:
                    min_frame_distance = distance  # 取最小的 frame 距離

            if min_frame_distance < float("inf"):
                total_distance += min_frame_distance  # 累積到總距離
                valid_comparisons += 1

        if valid_comparisons > 0:
            filename_distances[data["filename"]] = (total_distance / valid_comparisons, data.get("suggestion", "無"))  # 平均距離 & suggestion

    # 如果沒有任何有效的比對
    if not filename_distances:
        print("無有效的比對數據")
        return "無"

    # 找到累積距離最短的 filename
    best_filename = min(filename_distances, key=lambda x: filename_distances[x][0])
    print(f"最接近的檔案: {best_filename}, Suggestion: {filename_distances[best_filename][1]}")
    return filename_distances[best_filename][1]  # 回傳該 filename 的 suggestion

# **主執行函數**
if __name__ == "__main__":
    # **動態傳入 `filename`**
    dynamic_filename = "9_8(3D_trajectory_smoothed).json"  # 這裡可以換成不同影片名稱

    # 讀取檔案
    merged_dataset_path = "E:/git_repos/AI_Coach_Detection/KNN/merged_dataset.json"
    trajectory_path = f"E:/git_repos/AI_Coach_Detection/KNN/{dynamic_filename}"

    merged_dataset = load_json(merged_dataset_path)
    trajectory_data = load_json(trajectory_path)

    # 先將 trajectory_data 加入 merged_dataset
    merged_dataset.append({
        "filename": dynamic_filename,
        "level": "unknown",
        "suggestion": "無",
        "data": trajectory_data
    })

    # 進行標準化，只回傳 trajectory_data 的部分
    normalized_trajectory = normalize_data(merged_dataset, dynamic_filename)

    # 使用 KNN 找到最近的 suggestion
    best_suggestion = find_nearest_suggestion(normalized_trajectory, merged_dataset, dynamic_filename)

    # 儲存標準化後的 trajectory_data
    save_json(f"E:/git_repos/AI_Coach_Detection/KNN/normalized_{dynamic_filename}", normalized_trajectory)

    # **輸出標準化後的 trajectory_data 以及最近的 suggestion**
    print(json.dumps({"best_suggestion": best_suggestion}, indent=4, ensure_ascii=False))






