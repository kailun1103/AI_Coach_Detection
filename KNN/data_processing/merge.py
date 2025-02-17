import json

# 讀取 JSON 檔案
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 檔案路徑
junior_file = "E:/git_repos/AI_Coach_Detection/KNN/junior_labeled.json"
pro_file = "E:/git_repos/AI_Coach_Detection/KNN/pro_labeled.json"
output_file = "E:/git_repos/AI_Coach_Detection/KNN/merged_dataset.json"

# 讀取資料
junior_data = load_json(junior_file)
pro_data = load_json(pro_file)

# 合併數據
merged_data = junior_data + pro_data

# 儲存為新的 JSON
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(merged_data, outfile, indent=4, ensure_ascii=False)

print(f"合併完成！已儲存到 {output_file}")
