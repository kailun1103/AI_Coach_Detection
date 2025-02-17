import json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 讀取 JSON 檔案
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# 儲存 JSON 檔案
def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# 標準化座標數據（merged_dataset + trajectory_data）
def normalize_data(dataset, trajectory_filename):
    keys = ["nose", "left_shoulder", "right_shoulder", "right_elbow", "right_wrist", "left_knee", "right_knee"]

    # 取得所有幀的數據
    data_matrix = {key: [] for key in keys}
    all_frames = []
    trajectory_frames = None  # 存儲 trajectory_data 的幀數據

    # 收集所有幀
    for data in dataset:
        if data["filename"] == trajectory_filename:
            trajectory_frames = data["data"]  # 記錄 trajectory_data 的幀數據

        for frame in data["data"]:
            all_frames.append(frame)
            for key in keys:
                if key in frame and all(k in frame[key] for k in ["x", "y", "z"]):  # 確保該鍵值存在
                    data_matrix[key].append([frame[key]["x"], frame[key]["y"], frame[key]["z"]])

    # 初始化標準化器並對每個鍵值進行標準化
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

# KNN 計算最短距離，回傳最接近的檔案名稱與建議
def find_nearest_suggestion(trajectory_frames, merged_dataset, trajectory_filename):
    keys = ["nose", "left_shoulder", "right_shoulder", "right_elbow", "right_wrist", "left_knee", "right_knee"]
    filename_distances = {}

    for data in merged_dataset:
        if data["filename"] == trajectory_filename:
            continue  # 忽略自己，避免與自身進行比對

        total_distance = 0  # 累積所有幀的歐式距離
        valid_comparisons = 0  # 記錄有效的幀比對數

        # 確保 trajectory_frames 與 data["data"] 的長度一致，選擇較短的作為範圍
        min_frame_count = min(len(trajectory_frames), len(data["data"]))

        for i in range(min_frame_count):  # 逐一對應的幀計算距離
            traj_frame = trajectory_frames[i]
            ref_frame = data["data"][i]
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
                    distance += np.linalg.norm(traj_array - ref_array)  # 計算歐式距離

            # 累加當前幀的距離到總距離
            total_distance += distance
            valid_comparisons += 1

        if valid_comparisons > 0:
            filename_distances[data["filename"]] = (total_distance / valid_comparisons, data.get("suggestion", "None"))  # 平均距離 & 建議

    # 如果沒有任何有效的比對
    if not filename_distances:
        print("沒有有效的比對數據")
        return "None"

    # 找到累積距離最短的檔案名稱
    best_filename = min(filename_distances, key=lambda x: filename_distances[x][0])
    print(f"最接近的檔案: {best_filename}, 建議: {filename_distances[best_filename][1]}")
    return filename_distances[best_filename][1]  # 回傳該檔案的建議

# 主執行函數
if __name__ == "__main__":
    # 動態傳入檔案名稱
    dynamic_filename = "9_8(3D_trajectory_smoothed).json"  # 這裡可以換成不同的影片檔案名稱

    # 定義檔案路徑
    base_path = "E:/git_repos/AI_Coach_Detection/KNN"
    merged_dataset_path = os.path.join(base_path, "merged_dataset.json")
    trajectory_path = os.path.join(base_path, dynamic_filename)

    merged_dataset = load_json(merged_dataset_path)
    trajectory_data = load_json(trajectory_path)

    # 將 trajectory_data 加入 merged_dataset
    merged_dataset.append({
        "filename": dynamic_filename,
        "level": "unknown",
        "suggestion": "None",
        "data": trajectory_data
    })

    # 進行標準化，只回傳 trajectory_data
    normalized_trajectory = normalize_data(merged_dataset, dynamic_filename)

    # 使用 KNN 找到最近的建議
    best_suggestion = find_nearest_suggestion(normalized_trajectory, merged_dataset, dynamic_filename)

    # 建立輸出資料夾
    output_folder = os.path.join(base_path, f"suggestion_{dynamic_filename}")
    os.makedirs(output_folder, exist_ok=True)  # 如果資料夾不存在則建立

    # 儲存標準化後的 trajectory_data 到新資料夾
    normalized_trajectory_path = os.path.join(output_folder, f"normalized_{dynamic_filename}")
    save_json(normalized_trajectory_path, normalized_trajectory)

    # 儲存建議到 text 文件到新資料夾
    suggestion_path = os.path.join(output_folder, "suggestion.txt")
    with open(suggestion_path, "w", encoding="utf-8") as file:
        file.write(best_suggestion)

    # 輸出儲存的檔案路徑
    print(f"已儲存標準化後的軌跡數據: {normalized_trajectory_path}")
    print(f"已儲存建議文本檔案: {suggestion_path}")

    # 輸出標準化後的 trajectory_data 以及最近的建議
    print(json.dumps({"best_suggestion": best_suggestion}, indent=4, ensure_ascii=False))
