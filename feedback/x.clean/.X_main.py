import os
import json

def round_coordinates(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                data[key] = round_coordinates(value)
            elif isinstance(value, float):
                data[key] = round(value, 2)
    elif isinstance(data, list):
        data = [round_coordinates(item) if isinstance(item, (dict, list, float)) else item for item in data]
    return data
def main():
    # 定義資料夾路徑
    raw_data_folder = "./__data__/raw_data"
    standard_folder = "./__data__/standard_player"
    rookie_folder = "./__data__/rookie_player"
    # 確認資料夾是否存在，若無則建立
    os.makedirs(standard_folder, exist_ok=True)
    os.makedirs(rookie_folder, exist_ok=True)

    # 遍歷 raw_data 資料夾的所有檔案
    for file_name in os.listdir(raw_data_folder):
        # 確保只處理 .json 檔案
        if file_name.endswith(".json"):
            try:
                # 驗證檔案命名邏輯
                if "standard_raw" in file_name:
                    number = file_name.split("_")[-1].split(".")[0]  # 提取檔案編號
                    new_file_name = f"standard_{number}.json"
                    save_path = os.path.join(standard_folder, new_file_name)
                elif "rookie_raw" in file_name:
                    number = file_name.split("_")[-1].split(".")[0]  # 提取檔案編號
                    new_file_name = f"rookie_{number}.json"
                    save_path = os.path.join(rookie_folder, new_file_name)
                else:
                    # 不符合命名邏輯的檔案略過
                    continue

                file_path = os.path.join(raw_data_folder, file_name)
                
                # 讀取檔案
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                # 處理資料 (四捨五入座標到小數第二位)
                cleaned_data = round_coordinates(data)

                # 寫入新檔案
                with open(save_path, 'w', encoding='utf-8') as file:
                    json.dump(cleaned_data, file, ensure_ascii=False, indent=4)

                print(f"Processed and saved: {save_path}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

if __name__ == "__main__":
    main()
