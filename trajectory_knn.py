import json
import os
import numpy as np
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Load JSON file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Save JSON file
def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# Normalize coordinate data (merged_dataset + trajectory_data)
def normalize_data(dataset, trajectory_filename):
    keys = ["nose", "left_shoulder", "right_shoulder", "right_elbow", "right_wrist", "left_knee", "right_knee"]

    # Retrieve all frames data
    data_matrix = {key: [] for key in keys}
    all_frames = []
    trajectory_frames = None  # Store frames from trajectory_data

    # Collect all frames
    for data in dataset:
        if data["filename"] == trajectory_filename:
            trajectory_frames = data["data"]  # Store frames from trajectory_data

        for frame in data["data"]:
            all_frames.append(frame)
            for key in keys:
                if key in frame and all(k in frame[key] for k in ["x", "y", "z"]):  # Ensure key exists
                    data_matrix[key].append([frame[key]["x"], frame[key]["y"], frame[key]["z"]])

    # Initialize scaler and normalize each key
    scalers = {key: MinMaxScaler() for key in keys}
    normalized_data = {key: scalers[key].fit_transform(data_matrix[key]) for key in keys}

    # Update the original JSON data
    frame_idx = 0
    for data in dataset:
        for i, frame in enumerate(data["data"]):
            for key in keys:
                if key in frame:
                    frame[key]["x"], frame[key]["y"], frame[key]["z"] = normalized_data[key][frame_idx]
            frame_idx += 1

    return trajectory_frames  # Return normalized trajectory_data

# KNN calculates the shortest distance and returns the closest filename's suggestion
def find_nearest_suggestion(trajectory_frames, merged_dataset, trajectory_filename):
    keys = ["nose", "left_shoulder", "right_shoulder", "right_elbow", "right_wrist", "left_knee", "right_knee"]
    filename_distances = {}

    for data in merged_dataset:
        if data["filename"] == trajectory_filename:
            continue  # Skip itself to avoid self-comparison

        suggestion_text = data.get("suggestion", "")
        if "是否擊球:X" in suggestion_text:
            continue  # suggestion have "是否擊球:X" pass

        total_distance = 0  # Accumulate the Euclidean distance for all frames
        valid_comparisons = 0  # Count valid frame comparisons

        # Ensure trajectory_frames and data["data"] have the same length, choosing the shorter one
        min_frame_count = min(len(trajectory_frames), len(data["data"]))

        for i in range(min_frame_count):  # Compute distance for corresponding frames
            traj_frame = trajectory_frames[i]
            ref_frame = data["data"][i]
            distance = 0

            for key in keys:
                if key in traj_frame and key in ref_frame:
                    traj_point = traj_frame[key]
                    ref_point = ref_frame[key]

                    # Check if there are None values, skip if any exist
                    if any(v is None for v in traj_point.values()) or any(v is None for v in ref_point.values()):
                        continue

                    traj_array = np.array([traj_point["x"], traj_point["y"], traj_point["z"]])
                    ref_array = np.array([ref_point["x"], ref_point["y"], ref_point["z"]])
                    distance += np.linalg.norm(traj_array - ref_array)  # Compute Euclidean distance

            # Accumulate frame distance into total distance
            total_distance += distance
            valid_comparisons += 1

        if valid_comparisons > 0:
            filename_distances[data["filename"]] = (total_distance / valid_comparisons, data.get("suggestion", "None"))  # Average distance & suggestion

    # If no valid comparisons were found
    if not filename_distances:
        print("No valid comparison data")
        return "None"

    # Find the filename with the shortest cumulative distance
    best_filename = min(filename_distances, key=lambda x: filename_distances[x][0])
    print(f"Closest file: {best_filename}, Suggestion: {filename_distances[best_filename][1]}")
    return filename_distances[best_filename][1]  # Return the suggestion of the best-matched filename

if __name__ == "__main__":
    total_start_time = time.time()
    
    print(f"開始執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Dynamically pass filename
    dynamic_filename = "嘉洋__3(3D_trajectory_smoothed).json"
    
    # Define paths
    base_path = "C:/Users/d93xj/OneDrive/Desktop/AI_Coach_Detection"
    merged_dataset_path = ("merged_dataset.json")
    trajectory_path = dynamic_filename
    print('----')
    print(trajectory_path)
    print('----')

    # 計時：讀取資料
    load_start_time = time.time()
    merged_dataset = load_json(merged_dataset_path)
    trajectory_data = load_json(trajectory_path)
    load_time = time.time() - load_start_time
    print(f"\n讀取資料耗時: {load_time:.2f} 秒")

    # 計時：資料合併
    append_start_time = time.time()
    merged_dataset.append({
        "filename": dynamic_filename,
        "level": "unknown",
        "suggestion": "None",
        "data": trajectory_data
    })
    append_time = time.time() - append_start_time
    print(f"資料合併耗時: {append_time:.2f} 秒")

    # 計時：正規化
    norm_start_time = time.time()
    normalized_trajectory = normalize_data(merged_dataset, dynamic_filename)
    norm_time = time.time() - norm_start_time
    print(f"正規化耗時: {norm_time:.2f} 秒")

    # 計時：KNN 分析
    knn_start_time = time.time()
    best_suggestion = find_nearest_suggestion(normalized_trajectory, merged_dataset, dynamic_filename)
    knn_time = time.time() - knn_start_time
    print(f"KNN 分析耗時: {knn_time:.2f} 秒")

    # 計時：儲存結果
    save_start_time = time.time()
    
    output_folder = os.path.join(base_path, f"suggestion_{dynamic_filename}")
    os.makedirs(output_folder, exist_ok=True)

    normalized_trajectory_path = os.path.join(output_folder, f"normalized_{dynamic_filename}")
    save_json(normalized_trajectory_path, normalized_trajectory)

    suggestion_path = os.path.join(output_folder, "suggestion.txt")
    with open(suggestion_path, "w", encoding="utf-8") as file:
        file.write(best_suggestion)
        
    save_time = time.time() - save_start_time
    print(f"儲存結果耗時: {save_time:.2f} 秒")

    # 計算總執行時間
    total_time = time.time() - total_start_time
    print(f"\n總執行時間: {total_time:.2f} 秒")
    print(f"結束執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 輸出執行結果
    print("\n執行結果:")
    print(f"Saved normalized trajectory data: {normalized_trajectory_path}")
    print(f"Saved suggestion text file: {suggestion_path}")
    print(json.dumps({"best_suggestion": best_suggestion}, indent=4, ensure_ascii=False))