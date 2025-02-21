import os, itertools
from openai import OpenAI
import pandas as pd 
import single_feedback.prompt as prompt, single_feedback.model_config as model_config



class AIFeedback ():
    def __init__ (self):
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

    def response(self,my_motion,knn_feedback):
        
        print ("\nGernerating Response......\n")
        
        messages = [
            { "role": "system", 
            "content": self.INSTRUCTIONS },
            { "role": "system", 
            "content": self.DATADESCIRBE },
            {"role":"user",
            "content": f"""
            
                Rephrase {knn_feedback}, Describe the analysis results of each body part, 
                answer it in "suggestion" schema.
                
                Based on this {my_motion}, 
                infer in which frame section (e.g. 1-58) the issue described in the feedback occurs.
                answer it in "frame" schema.
                
                """} 
        ]
        
        # ---RESPONSE---
        completion = self.model_config(messages)
        response = completion.choices[0].message.content
        messages.append({ "role": "assistant", "content": response })      
        print ("網球教練Frank: " + response)
        
        # ---SAVE FEEDBACK---
        ai_feedback =  [msg for msg in messages if msg["role"] == "assistant"]
        return ai_feedback 

    def conclude(self,ai_feedback):
        print ("\nGernerating Conclusion......\n")
        
        messages = [
            { "role": "system", 
            "content": self.INSTRUCTIONS },
            {"role":"user",
            "content": f"""
            
                Based on the previous {ai_feedback}, 
                give me a clear and easy-to-understand conclusion about my swing motion in Traditional Chinese, 
                as if you were a coach giving friendly, spoken feedback.
                
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
        
    def find_filepath(self, target_name, filename, max_count=10):
        filepaths = []
        for i in range(1, max_count + 1):
            file_path = f"./__data__/prod/{target_name}_trajectory/trajectory_{i}/{filename}"
            if not os.path.exists(file_path):
                break
            filepaths.append(file_path)
        return filepaths

    def main(self):
        ai_feedback = []
        target_name = "player01"
        
        json_filepaths = self.find_filepath(target_name, "trajectory.json")
        txt_filepaths = self.find_filepath(target_name, "suggestion.txt")
        
        for j, k in zip(json_filepaths, txt_filepaths):
            motion = pd.read_json(j)
            knn = pd.read_fwf(k)
            motion = self.process_data(motion)
            response = self.response(motion, knn)
            ai_feedback.append(response)
        
        self.conclude(ai_feedback)
    