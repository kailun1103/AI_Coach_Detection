import os
import pandas as pd
import json
import time
import single_feedback.prompt as prompt
import single_feedback.model_config as model_config
from openai import OpenAI
from open_ai_key import api_key

# --- 設定 API 參數與載入 Prompt 與模型設定 ---
client = OpenAI(api_key=api_key)
MODEL = model_config.MODEL
TEMPERATURE = model_config.TEMPERATURE
MAX_TOKENS = model_config.MAX_TOKENS
FREQUENCY_PENALTY = model_config.FREQUENCY_PENALTY
PRESENCE_PENALTY = model_config.PRESENCE_PENALTY
TOP_P = model_config.TOP_P

INSTRUCTIONS = prompt.INSTRUCTIONS
DATADESCIRBE = prompt.DATADESCIRBE

def create_chat_completion(messages):
    """
    以給定的 messages 呼叫 OpenAI ChatCompletion
    回傳產生的 completion 結果
    """
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

def generate_feedback(json_filepath, txt_filepath):
    """
    讀取 JSON (運動軌跡) 與 KNN 結果(txt)，並綜合兩者資訊產出 GPT 回饋
    最後將結果輸出為 _gpt_feedback.json 檔
    """
    # 讀取運動軌跡資料與 KNN 回饋
    my_motion = pd.read_json(json_filepath)
    knn_feedback = pd.read_csv(txt_filepath, header=None).iloc[0, 0]

    # 初始化 messages 列表
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "system", "content": DATADESCIRBE},
    ]

    # 如果 knn_feedback 為特定正向回饋訊息
    if knn_feedback == "頭:沒問題!、肩膀:沒問題!、手碗:沒問題!、手肘:沒問題!、膝蓋:沒問題!、是否擊球:是、其他:無":
        knn_response = "沒有觀察到顯著問題，請繼續保持！"
        frame_response = "0-0"

        # 將 frame 與建議回饋一起附加到 messages 中
        messages.append({"role": "assistant", "content": frame_response})
        messages.append({"role": "assistant", "content": knn_response})

    else:
        # 第一次讓 GPT 根據 KNN Feedback 產生中文敘述
        messages.append({
            "role": "user",
            "content": f"""
                observe analysis results: {knn_feedback}, 
                Rephrase the analysis results of each body part in 1 sentence
            """
        })
        knn_completion = create_chat_completion(messages)
        knn_response = knn_completion.choices[0].message.content

        # 讓 GPT 根據 json 內容推測大致在第幾幀區間會出現問題
        messages.append({
            "role": "user",
            "content": f"""
                Based on this {my_motion}, 
                Speculate in which frame section the issue described in the feedback occurs. 
                Please provide a broader frame range covering more frames (e.g., a range of at least 15 frames), 
                and You MUST respond with a numeric range only, in the format "number-number" (e.g., "13-24"), 
                containing only digits and a hyphen, with no additional text or formatting.
            """
        })
        frame_completion = create_chat_completion(messages)
        frame_response = frame_completion.choices[0].message.content

        # 將數字範圍與 knn_response 加入到 messages (可以用於後續檢視或除錯)
        messages.append({"role": "assistant", "content": frame_response})
        messages.append({"role": "assistant", "content": knn_response})

    # 處理換行符號
    frame_response = frame_response.replace("\n", "")
    knn_response = knn_response.replace("\n", "")

    # 構造 JSON 格式回傳結果
    ai_feedback = {
        "problem_frame": frame_response,
        "suggestion": knn_response,
    }

    print(ai_feedback)

    # 輸出檔案路徑 (以原檔案名稱 + "_gpt_feedback.json")
    output_filepath = json_filepath.replace('(3D_trajectory_smoothed)_only_swing.json', '_gpt_feedback.json')
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(ai_feedback, f, ensure_ascii=False, indent=2)

    return output_filepath


if __name__ == "__main__":
    json_path = "嘉洋__3(3D_trajectory_smoothed).json"
    txt_path = "嘉洋__3_knn_feedback.txt"

    # 開始計時
    start_time = time.time()

    # 產生並輸出回饋
    output_filepath = generate_feedback(json_path, txt_path)

    # 結束計時
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("AI Feedback:")
    print(f"Processing time: {elapsed_time:.2f} seconds")