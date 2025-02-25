import os
import pandas as pd
from openai import OpenAI
import single_feedback.prompt as prompt
import single_feedback.model_config as model_config
import json
import re
from open_ai_key import api_key

# --- 全域設定 ---
client = OpenAI(api_key=api_key)

MODEL = model_config.MODEL
TEMPERATURE = model_config.TEMPERATURE
MAX_TOKENS = model_config.MAX_TOKENS
FREQUENCY_PENALTY = model_config.FREQUENCY_PENALTY
PRESENCE_PENALTY = model_config.PRESENCE_PENALTY
TOP_P = model_config.TOP_P

INSTRUCTIONS = prompt.INSTRUCTIONS
DATADESCIRBE = prompt.DATADESCIRBE

# --- 模型呼叫函式 ---
def model_config_call(messages):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion

def find_and_format_feedback_jsons(folder_path):   
    # 用於存儲整合後的結果
    formatted_results = []
    
    # 計數器，用於追蹤找到的檔案數量
    found_files = 0
    
    # 遍歷資料夾及其子資料夾
    for root, dirs, files in os.walk(folder_path):
        # 篩選出包含 '_feedback.json' 的檔案
        feedback_files = [f for f in files if '_feedback.json' in f]
        
        # 對檔案進行排序，確保按照序號順序處理
        # 假設檔案名格式為 '..._N_feedback.json'，其中N是數字
        feedback_files.sort(key=lambda x: int(re.search(r'_(\d+)_', x).group(1)) if re.search(r'_(\d+)_', x) else 0)
        
        # 如果在當前資料夾中找到了符合條件的檔案
        for file in feedback_files:
            found_files += 1
            file_path = os.path.join(root, file)
            
            # 從檔案名或路徑獲取軌跡編號
            trajectory_number = re.search(r'trajectory__(\d+)', file_path)
            trajectory_num = int(trajectory_number.group(1)) if trajectory_number else found_files
            
            # 計算幀號範圍，這裡只是示例，您需要根據實際情況調整
            # 這裡假設每個軌跡對應的幀號為 30+軌跡編號
            frame_start = 30 + trajectory_num
            frame_end = frame_start + 1 if trajectory_num != 1 else frame_start
            frame_range = f"{frame_start}-{frame_end}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 提取problem_frame和suggestion
                problem_frame = data.get('problem_frame', '0-0')
                suggestion = data.get('suggestion', '')
                
                # 如果problem_frame是'0-0'且有具體描述，則使用計算出的幀號範圍
                if problem_frame == '0-0' and suggestion != "沒有觀察到顯著問題，請繼續保持！":
                    problem_frame = frame_range
                
                # 創建格式化的結構
                formatted_item = [
                    {'role': 'assistant', 'content': problem_frame},
                    {'role': 'assistant', 'content': suggestion}
                ]
                
                # 根據是否有問題調整提示文字
                if suggestion == "沒有觀察到顯著問題，請繼續保持！":
                    knn_message = "根據K-Nearest Neighbor分析結果，您的頭部、肩膀、手腕、手肘和膝蓋的揮拍動作都沒有問題，並且成功擊球，未發現其他異常。"
                    formatted_item[1]['content'] = knn_message
                
                formatted_results.append(formatted_item)
    
    return formatted_results

# --- 資料處理函式 ---
def process_data(motion):
    processed = []
    for index, row in motion.iterrows():
        new_item = {}
        new_item["frame"] = row.get("frame")
        right_wrist = row.get("right_wrist", {})
        processed_right_wrist = {}
        for key in ["x", "y", "z"]:
            value = right_wrist.get(key)
            if isinstance(value, (int, float)):
                processed_right_wrist[key] = round(value, 2)
            else:
                processed_right_wrist[key] = value
        new_item["right_wrist"] = processed_right_wrist
        new_item["tennis_ball_hit"] = row.get("tennis_ball_hit")
        angle = row.get("tennis_ball_angle")
        new_item["tennis_ball_angle"] = round(angle, 2) if isinstance(angle, (int, float)) else angle
        processed.append(new_item)
    return processed

def find_filepath(filename, max_count=3):
    filepaths = []
    for i in range(1, max_count + 1):
        file_path = f"trajectory/嘉洋__trajectory/trajectory__{i}/{filename}{i}(3D_trajectory_smoothed).json"
        print(file_path)
        if not os.path.exists(file_path):
            print("找不到檔案，結束搜尋")
            break
        filepaths.append(file_path)
    return filepaths

def find_filepathtxt(filename, max_count=3):
    filepaths = []
    for i in range(1, max_count + 1):
        file_path = f"trajectory/嘉洋__trajectory/trajectory__{i}/{filename}{i}_knn_feedback.txt"
        print(file_path)
        if not os.path.exists(file_path):
            print("找不到檔案，結束搜尋")
            break
        filepaths.append(file_path)
    return filepaths

# --- 根據輸入內容產生 ai_feedback ---
def generate_ai_feedback(my_motion, knn_feedback):
    # 此處僅直接將 knn 的回饋文字帶入，
    # 並以 "0-0" 當作 frame 參考值（可依實際需求修改）
    frame_response = "0-0"
    knn_response = knn_feedback

    # 組成 assistant 的回饋訊息列表
    ai_feedback = [
        {"role": "assistant", "content": frame_response},
        {"role": "assistant", "content": knn_response}
    ]

    # 輸出供檢查用
    print(f'{{\n  "frame": "{frame_response}",\n  "suggestion": "{knn_response}"\n}}')
    return ai_feedback

# --- 最後結論函式 ---
def conclude(ai_feedback):
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": f"""
        Based on the previous {ai_feedback}, 
        You will see a KNN analysis feedback on different body parts of a tennis beginner during various swing attempts.
        For each body part listed below, conclude the issue in one sentence. Finally, provide one sentence of advice to help improve the beginner's swing.
        Body parts: Head, Shoulders, Wrists, Elbows, Knees.
        """}
    ]
    completion = model_config_call(messages)
    response = completion.choices[0].message.content
   
    return response


if __name__ == "__main__":

    folder_path = 'trajectory/嘉洋__trajectory'
    
    # 執行主函數並格式化結果
    results = find_and_format_feedback_jsons(folder_path)
    
    # 呼叫 conclude，將所有 ai_feedback 傳入，並回傳最終結論
    final_conclusion = conclude(results)

    print(final_conclusion)

