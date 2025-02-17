import os
import json

# 定義要處理的檔案資料夾路徑
input_folder = 'E:/git_repos/AI_Coach_Detection/KNN/Junior_dataset'  # 請替換成您的輸入檔案資料夾路徑
output_folder = 'E:/git_repos/AI_Coach_Detection/KNN/Junior_Labeled_Dataset'  # 請替換成您想存放輸出檔案的資料夾路徑

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 遍歷輸入資料夾內的所有 JSON 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.json'):  # 確保只處理 JSON 檔案
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, f'{file_name}')
        
        # 讀取原始 JSON 文件
        with open(input_file_path, 'r') as input_file:
            data = json.load(input_file)
        
        # 添加 "level": "pro" 在外層
        output_data = {
            "level": "junior",
            "data": data
        }
        
        # 保存修改後的 JSON 文件
        with open(output_file_path, 'w') as output_file:
            json.dump(output_data, output_file, indent=4)
        
        print(f'處理完成: {file_name} -> {output_file_path}')

print("所有檔案處理完成！")