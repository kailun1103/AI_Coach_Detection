import json

file_name = "standard"
file_path = f"./{file_name}.json"

# 讀取 JSON 檔案
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 移除 'right_wrist_vector' 欄位
for entry in data:
    if 'right_wrist_vector' in entry:
        del entry['right_wrist_vector']

# 將清理後的資料寫回檔案
with open(f'{file_name}_cleaned.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("清理完成，已儲存至'chu_cleaned.json'")
