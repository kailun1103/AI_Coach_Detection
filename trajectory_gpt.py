import os, itertools
from openai import OpenAI
import pandas as pd 
import single_feedback.prompt as prompt, single_feedback.model_config as model_config
from open_ai_key import api_key


class AIFeedback ():
    def __init__ (self):
        # ---CLIENT---
        self.client = OpenAI(api_key=api_key)
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
  
    def model_config(self,messages):
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
        # 初始化 messages 列表
        messages = [
            {"role": "system", "content": self.INSTRUCTIONS},
            {"role": "system", "content": self.DATADESCIRBE},
        ]

        # 若 knn_feedback 為特定正向回饋訊息
        if knn_feedback == "頭:沒問題!、肩膀:沒問題!、手碗:沒問題!、手肘:沒問題!、膝蓋:沒問題!、其他:沒問題!":
            knn_response = "沒有觀察到顯著問題，請繼續保持！"
            frame_response = "0-0"

            messages.append({"role": "assistant", "content": frame_response})
            messages.append({"role": "assistant", "content": knn_response})

            print(f'{{\n  "frame": "{frame_response}",\n  "suggestion": "{knn_response}"\n}}')

            # ---SAVE FEEDBACK---
            ai_feedback = [msg for msg in messages if msg["role"] == "assistant"]
            return ai_feedback

        # 否則，進行一般的回應流程
        else:
            messages.append({
                "role": "user",
                "content": f"""
                    observe analysis results: {knn_feedback}, 
                    Rephrase the analysis results of each body part in 1 sentence
                """
            })
            knn_completion = self.model_config(messages)
            knn_response = knn_completion.choices[0].message.content

            messages.append({
                "role": "user",
                "content": f"""
                    Based on this {my_motion}, 
                    Speculate in which frame section the issue described in the feedback occurs, 
                    answer will only be in format "number"-"number" and nothing more, for example:13-24
                """
            })
            frame_completion = self.model_config(messages)
            frame_response = frame_completion.choices[0].message.content

            messages.append({"role": "assistant", "content": frame_response})
            messages.append({"role": "assistant", "content": knn_response})

            print(f'{{\n  "frame": "{frame_response}",\n  "suggestion": "{knn_response}"\n}}')

            # ---SAVE FEEDBACK---
            ai_feedback = [msg for msg in messages if msg["role"] == "assistant"]
            return ai_feedback

    def conclude(self,ai_feedback): 

        print('ai_feedback')
        print(ai_feedback)
        print('ai_feedback')

        messages = [
            { "role": "system", 
            "content": self.INSTRUCTIONS },
            {"role":"user",
            "content": f"""
    
                Based on the previous {ai_feedback}, 
                You will see a KNN analysis feedback on different body parts of a tennis beginner during various swing attempts.
                For each body part listed below, conclude the issue in one sentence. Finally, provide one sentence of advice to help improve the beginner's swing.
                Body parts: Head, Shoulders, Wrists, Elbows, Knees.
    
                """
            } 
        ]
        completion = self.model_config(messages)
        
        response = completion.choices[0].message.content
        print ("---CONCLUSION---")
        print ("\n",response)
    
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
        
    def find_filepath(self, filename, max_count=3):
        filepaths = []
        for i in range(1, max_count + 1):
            file_path = f"trajectory/嘉洋__trajectory/trajectory__{i}/{filename}{i}(3D_trajectory_smoothed).json"
            print(file_path)
            if not os.path.exists(file_path):
                print(1)
                break
            filepaths.append(file_path)
        return filepaths
    
    def find_filepathtxt(self, filename, max_count=3):
        filepaths = []
        for i in range(1, max_count + 1):
            file_path = f"trajectory/嘉洋__trajectory/trajectory__{i}/{filename}{i}_knn_feedback.txt"
            print(file_path)
            if not os.path.exists(file_path):
                print(2)
                break
            filepaths.append(file_path)
        return filepaths

    def main(self):
        ai_feedback = []
        
        json_filepaths = self.find_filepath("嘉洋__")
        txt_filepaths = self.find_filepathtxt("嘉洋__")
        
        for j, k in zip(json_filepaths, txt_filepaths):
            motion = pd.read_json(j)
            knn = pd.read_csv(k, header=None).iloc[0, 0]  # 讀取第一行第一個欄位

            motion = self.process_data(motion)
            response = self.response(motion, knn)
            ai_feedback.append(response)
        
        self.conclude(ai_feedback)

if __name__ == "__main__":
    feedback = AIFeedback()
    feedback.main()