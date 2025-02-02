import os
import json
from collections import OrderedDict

def insert_filename_before_level(obj, filename):
    """
    在 JSON 字典內的 "level": "pro" 之前插入 "filename" 欄位
    """
    if isinstance(obj, dict):
        new_obj = OrderedDict()
        inserted = False
        for key, value in obj.items():
            if key == "level" and value == "pro" and not inserted:
                new_obj["filename"] = filename  # 插入檔名
                inserted = True
            new_obj[key] = value
        if not inserted:
            new_obj["filename"] = filename  # 如果沒有 "level": "pro"，則放最後
        return new_obj
    return obj

def add_filename_to_json(data, filename):
    """
    遍歷 JSON 數據，並對每個字典插入檔名
    """
    if isinstance(data, list):
        return [insert_filename_before_level(item, filename) for item in data]
    elif isinstance(data, dict):
        return insert_filename_before_level(data, filename)
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
                    data_with_filename = add_filename_to_json(data, filename)  # 添加檔名資訊
                    if isinstance(data_with_filename, list):
                        merged_data.extend(data_with_filename)
                    else:
                        merged_data.append(data_with_filename)
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失敗：{file_path}，錯誤：{e}")

    # 將合併的 JSON 儲存到新的檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"合併完成！輸出檔案：{output_file}")

# 使用方式
input_folder = "E:/git_repos/AI_Coach_Detection/KNN/Labeled_Dataset"  # 設定你的 JSON 資料夾路徑
output_file = "E:/git_repos/AI_Coach_Detection/KNN/merged.json"  # 輸出的 JSON 檔案名稱
merge_json_files(input_folder, output_file)


