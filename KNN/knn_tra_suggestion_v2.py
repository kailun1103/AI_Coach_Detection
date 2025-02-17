import json
import os
import numpy as np
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

# Main execution function
if __name__ == "__main__":
    # Dynamically pass filename
    dynamic_filename = "9_8(3D_trajectory_smoothed).json"  # This can be changed to different video filenames

    # Define paths
    base_path = "E:/git_repos/AI_Coach_Detection/KNN"
    merged_dataset_path = os.path.join(base_path, "merged_dataset.json")
    trajectory_path = os.path.join(base_path, dynamic_filename)

    merged_dataset = load_json(merged_dataset_path)
    trajectory_data = load_json(trajectory_path)

    # Append trajectory_data to merged_dataset
    merged_dataset.append({
        "filename": dynamic_filename,
        "level": "unknown",
        "suggestion": "None",
        "data": trajectory_data
    })

    # Perform normalization, only return trajectory_data
    normalized_trajectory = normalize_data(merged_dataset, dynamic_filename)

    # Use KNN to find the nearest suggestion
    best_suggestion = find_nearest_suggestion(normalized_trajectory, merged_dataset, dynamic_filename)

    # Create output folder
    output_folder = os.path.join(base_path, f"suggestion_{dynamic_filename}")
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    # Save the normalized trajectory_data inside the new folder
    normalized_trajectory_path = os.path.join(output_folder, f"normalized_{dynamic_filename}")
    save_json(normalized_trajectory_path, normalized_trajectory)

    # Save suggestion as a text file inside the new folder
    suggestion_path = os.path.join(output_folder, "suggestion.txt")
    with open(suggestion_path, "w", encoding="utf-8") as file:
        file.write(best_suggestion)

    # Output the saved paths
    print(f"Saved normalized trajectory data: {normalized_trajectory_path}")
    print(f"Saved suggestion text file: {suggestion_path}")

    # Output normalized trajectory_data and the closest suggestion
    print(json.dumps({"best_suggestion": best_suggestion}, indent=4, ensure_ascii=False))

