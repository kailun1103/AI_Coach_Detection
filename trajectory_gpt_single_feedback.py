import os
from openai import OpenAI
import pandas as pd 
import single_feedback.prompt as prompt, single_feedback.model_config as model_config
import time

class AIFeedback():
    def __init__(self):
        # ---CLIENT---
        self.client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))
        # ---MODEL CONFIG---
        self.MODEL = model_config.MODEL
        self.TEMPERATURE = model_config.TEMPERATURE
        self.MAX_TOKENS = model_config.MAX_TOKENS
        self.FREQUENCY_PENALTY = model_config.FREQUENCY_PENALTY
        self.PRESENCE_PENALTY = model_config.PRESENCE_PENALTY
        self.MAX_CONTEXT_QUESTIONS = model_config.MAX_CONTEXT_QUESTIONS
        self.TOP_P = model_config.TOP_P
        # ---MODEL PROMPT---
        self.INSTRUCTIONS = prompt.INSTRUCTIONS 
        self.DATADESCIRBE = prompt.DATADESCIRBE
  
    def model_config(self, messages):
        completion = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            top_p=self.TOP_P,
            frequency_penalty=self.FREQUENCY_PENALTY,
            presence_penalty=self.PRESENCE_PENALTY,
        )   
        return completion

    def response(self, my_motion, knn_feedback):
        print("\nGernerating Response......")
        
        messages = [
            {"role": "system", "content": self.INSTRUCTIONS},
            {"role": "system", "content": self.DATADESCIRBE},
            {"role": "user", "content": f"""
                Rephrase {knn_feedback}, Describe the analysis results of each body part, 
                answer it in "suggestion" schema.
                
                Based on this {my_motion}, 
                infer in which frame section (e.g. 1-58) the issue described in the feedback occurs.
                answer it in "frame" schema.
            """} 
        ]
        
        completion = self.model_config(messages)
        response = completion.choices[0].message.content
        print("網球教練Frank: " + response)
        return response

    def process_data(self, motion):
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

    def main(self, json_path, txt_path):
        start_time = time.time()
        
        print(f"開始處理檔案...")
        print(f"JSON檔案: {json_path}")
        print(f"TXT檔案: {txt_path}")
        
        # 讀取檔案
        motion = pd.read_json(json_path)
        with open(txt_path, 'r', encoding='utf-8') as f:
            knn = f.read()
        
        # 處理資料
        motion = self.process_data(motion)
        
        # 生成回饋
        response = self.response(motion, knn)
        
        end_time = time.time()
        print(f"\n處理完成，耗時: {end_time - start_time:.2f} 秒")
        
        return response

# 使用範例
if __name__ == "__main__":
    feedback = AIFeedback()
    json_file = "張凱倫__2(3D_trajectory_smoothed).json"
    txt_file = "張凱倫__2(3D_trajectory_smoothed).txt"
    feedback.main(json_file, txt_file)