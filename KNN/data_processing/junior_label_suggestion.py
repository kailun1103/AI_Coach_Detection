import os
import json
from collections import OrderedDict

def insert_suggestion_after_level(obj, filename):
    """
    在 JSON 字典內插入 "filename"，然後 "level": "junior"，最後 "suggestion"
    """
    if isinstance(obj, dict):
        new_obj = OrderedDict()
        new_obj["filename"] = filename  # 先插入檔名
        inserted = False
        for key, value in obj.items():
            new_obj[key] = value
            if key == "level" and value == "junior" and not inserted:
                new_obj["suggestion"] = "頭:O、肩膀:O、手碗:O、手肘:O、膝蓋:O、是否擊球:O、其他:無"
                inserted = True
        if not inserted:
            new_obj["suggestion"] = "頭:O、肩膀:O、手碗:O、手肘:O、膝蓋:O、是否擊球:O、其他:無"  # 如果沒有 "level": "pro"，則放最後
        return new_obj
    return obj

def add_filename_to_json(data, filename):
    """
    遍歷 JSON 數據，並對每個字典插入建議與檔名
    """
    if isinstance(data, list):
        return [insert_suggestion_after_level(item, filename) for item in data]
    elif isinstance(data, dict):
        return insert_suggestion_after_level(data, filename)
    return data

def merge_json_files(input_folder, output_file):
    merged_data = []  # 存儲合併的 JSON 內容

    # 獲取資料夾內所有 JSON 檔案
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    data_with_suggestion = add_filename_to_json(data, filename)  # 添加建議與檔名資訊
                    if isinstance(data_with_suggestion, list):
                        merged_data.extend(data_with_suggestion)
                    else:
                        merged_data.append(data_with_suggestion)
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失敗：{file_path}，錯誤：{e}")

    # 將合併的 JSON 儲存到新的檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"合併完成！輸出檔案：{output_file}")

# 使用方式
input_folder = "E:/git_repos/AI_Coach_Detection/KNN/Junior_Labeled_Dataset"  # 設定你的 JSON 資料夾路徑
output_file = "E:/git_repos/AI_Coach_Detection/KNN/junior_labeled.json"  # 輸出的 JSON 檔案名稱
merge_json_files(input_folder, output_file)