import json

# ---------------------------------------------------------------------------
# ------------------------------ Side Function ------------------------------
# ---------------------------------------------------------------------------

# Function to load JSON from local file
def load_json_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to save the comparison result to a file
def save_to_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Function to calculate differences between corresponding points in two datasets
def calculate_differences(json1, json2):
    differences = []
    if len(json1) != len(json2):
        print("Error: The two JSON files have different numbers of entries.")
        return None

    for i in range(len(json1)):
        if json1[i]["frame"] == json2[i]["frame"]:
            diff = {
                "frame": json1[i]["frame"],
                "right_wrist": {
                    "dx": json1[i]["right_wrist"]["x"] - json2[i]["right_wrist"]["x"],
                    "dy": json1[i]["right_wrist"]["y"] - json2[i]["right_wrist"]["y"]
                },
                "right_shoulder": {
                    "dx": json1[i]["right_shoulder"]["x"] - json2[i]["right_shoulder"]["x"],
                    "dy": json1[i]["right_shoulder"]["y"] - json2[i]["right_shoulder"]["y"]
                },
                "right_hip": {
                    "dx": json1[i]["right_hip"]["x"] - json2[i]["right_hip"]["x"],
                    "dy": json1[i]["right_hip"]["y"] - json2[i]["right_hip"]["y"]
                }
            }
            differences.append(diff)
        else:
            print(f"Warning: Frames do not match for entry {i} (Frame {json1[i]['frame']} vs {json2[i]['frame']})")
    
    return differences

# ---------------------------------------------------------------------------
# ------------------------------ Variable ------------------------------
# ---------------------------------------------------------------------------
# Load JSON data from local files
json1_path = './/standard_cleaned.json'
json2_path = './standard_cleaned.json'
output_file = './Camparision/Clean_compare.json'


JSON1 = load_json_from_file(json1_path)
JSON2 = load_json_from_file(json2_path)

# Compare and calculate differences
differences = calculate_differences(JSON1, JSON2)

# ---------------------------------------------------------------------------
# ------------------------------ Main Function ------------------------------
# ---------------------------------------------------------------------------

# If there are valid differences, save them to a file
if differences is not None:
    save_to_file(differences, output_file)
    print(f"\nComparison results saved to {output_file}\n")
else:
    print("No comparison results to save due to error or mismatch.")
