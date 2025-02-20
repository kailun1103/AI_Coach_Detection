import os
from openai import OpenAI
from dotenv import load_dotenv
from colorama import Fore, Back, Style
import pandas as pd 
import single_feedback.prompt as prompt, single_feedback.model_config as model_config


load_dotenv()

# class AIFeedback ():
#     def __init__ (self):
#         # ---CLIENT---
#         self.client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))
#         # ---MODEL CONFIG---
#         self.MODEL = model_config.MODEL
#         self.TEMPERATURE = model_config.TEMPERATURE
#         self.MAX_TOKENS = model_config.MAX_TOKENS
#         self.FREQUENCY_PENALTY = model_config.FREQUENCY_PENALTY
#         self.PRESENCE_PENALTY = model_config.PRESENCE_PENALTY
#         self.MAX_CONTEXT_QUESTIONS = model_config.MAX_CONTEXT_QUESTIONS
#         self.TOP_P = model_config.TOP_P
#         # ---MODEL PROMPT---
#         self.INSTRUCTIONS = prompt.INSTRUCTIONS 
#         self.PROMPT_1 = prompt.PROMPT_1
  
#     def model_config(self,messages):
#         completion = self.client.chat.completions.create(
#             model=self.MODEL,
#             messages=messages,
#             temperature=self.TEMPERATURE,
#             max_tokens=self.MAX_TOKENS,
#             top_p=self.TOP_P,
#             frequency_penalty=self.FREQUENCY_PENALTY,
#             presence_penalty=self.PRESENCE_PENALTY,
#         )   
#         return completion

#     def response(self,my_motion,coach_motion):
#         print ("\nGernerating Response......\n")
        
#         messages = [
#             { "role": "system", 
#             "content": self.INSTRUCTIONS },
#             { "role": "system", 
#             "content": self.PROMPT_1 },
#             {"role":"user",
#             "content": f"here is the json file of my tennis swing motion:{my_motion}, Please compare this with coach's motion {coach_motion} and base on the difference, give me some advice, let me know how to improve my swing motion."} 
#         ]
        
#         # ---RESPONSE---
#         completion = self.model_config(messages)
#         response = completion.choices[0].message.content
#         messages.append({ "role": "assistant", "content": response })      
#         print (Fore.CYAN + Style.BRIGHT + "網球教練Frank: " + Style.NORMAL + response)
#         # ---SAVE FEEDBACK---
#         ai_feedback =  [msg for msg in messages if msg["role"] == "assistant"]
#         return ai_feedback 

#     def conclude(self,ai_feedback):
#         print ("\nGernerating Conclusion......\n")
        
#         messages = [
#             { "role": "system", 
#             "content": self.INSTRUCTIONS },
#             {"role":"user",
#             "content": f"Based on the previous {ai_feedback}, give me a clear and easy-to-understand conclusion about my swing motion in Traditional Chinese, as if you were a coach giving friendly, spoken feedback."} 
#         ]
#         completion = self.model_config(messages)
        
#         response = completion.choices[0].message.content
#         print ("---CONCLUSION---")
#         print ("\n",response)
        
#     def test():
#         pass
    
#     def main(self):
#         target = [f"junior_9_{i}_cleaned" for i in range(3)]
#         target_filepaths = [f"./__data__/junior/{t}.json" for t in target]
#         ai_feedback = []
        
#         for t in target_filepaths:
#             pro = "pro_cleaned"
#             pro_filepath = f"./__data__/pro/{pro}.json"
            
#             my_motion = pd.read_json(t)
#             coach_motion = pd.read_json(pro_filepath) 
            
#             response = self.response(my_motion, coach_motion)
#             ai_feedback.append(response)
        
#         self.conclude(ai_feedback)   
    
class KNNFeedback():
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
        self.INSTRUCTIONS = prompt.KNN_INSTRUCTIONS 
        self.PROMPT_1 = prompt.PROMPT_1
  
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
    
    def response(self,knn_feedback):
        print ("\nGernerating Response......\n")
        
        messages = [
            { "role": "system", 
            "content": self.INSTRUCTIONS },
            {"role":"user",
            "content": f"here is the feedback of KNN to my tennis swing motion:{knn_feedback}, Please sum it up for me"} 
        ]
        
        # ---RESPONSE---
        completion = self.model_config(messages)
        response = completion.choices[0].message.content
        messages.append({ "role": "assistant", "content": response })      
        print (Fore.CYAN + Style.BRIGHT + "網球教練Frank: " + Style.NORMAL + response)
        
        # ---SAVE FEEDBACK---
        ai_feedback =  [msg for msg in messages if msg["role"] == "assistant"]
        return ai_feedback 

    def conclude(self,ai_feedback):
        print ("\nGernerating Conclusion......\n")
        
        messages = [
            { "role": "system", 
            "content": self.INSTRUCTIONS },
            {"role":"user",
            "content": f"Based on the previous {ai_feedback}, give me a clear and easy-to-understand conclusion about my swing motion in Traditional Chinese, as if you were a coach giving friendly, spoken feedback."} 
        ]
        completion = self.model_config(messages)
        
        response = completion.choices[0].message.content
        print ("---CONCLUSION---")
        print ("\n",response)

    def main(self):
        target = [f"player01-{i}" for i in range(1,4)]
        target_filepaths = [f"./__data__/knn_feedbacks/{t}.txt" for t in target]
        ai_feedback = []
        
        for t in target_filepaths:
            knn_feedback = pd.read_fwf(t)
        
            response = self.response(knn_feedback)
            ai_feedback.append(response)
        
        self.conclude(ai_feedback) 


if __name__ == "__main__":
    # main = AIFeedback()
    main = KNNFeedback()
    main.main()