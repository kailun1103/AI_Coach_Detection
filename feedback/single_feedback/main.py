import os
from openai import OpenAI
from dotenv import load_dotenv
from colorama import Fore, Back, Style
import pandas as pd 
import single_feedback.config as config

load_dotenv()
# -----------------------------
# --------Model setting--------
# -----------------------------
client = config.client
TEMPERATURE = config.TEMPERATURE
MAX_TOKENS = config.MAX_TOKENS
FREQUENCY_PENALTY = config.FREQUENCY_PENALTY
PRESENCE_PENALTY = config.PRESENCE_PENALTY
MAX_CONTEXT_QUESTIONS = config.MAX_CONTEXT_QUESTIONS
TOP_P = config.TOP_P
# -----------------------------
# --------Prompt setting--------
# -----------------------------
INSTRUCTIONS = config.INSTRUCTIONS 
PROMPT_1 = config.PROMPT_1
# -----------------------------
# --------File setting--------
# -----------------------------
pro = config.pro
pro_filepath = config.pro_filepath
# -----------------------------
# -------------DEF-------------
# -----------------------------
def system_message(INSTRUCTIONS, my_motion, coach_motion):    
    print ("\nGernerating Response......\n")
    messages = [
        { "role": "system", 
          "content": INSTRUCTIONS },
        { "role": "system", 
          "content": PROMPT_1 },
        {"role":"user",
         "content": f"here is the json file of my tennis swing motion:{my_motion}. Please compare this with coach's motion {coach_motion}, and base on the difference give  me some advice, let me know how to improve my swing motion."} 
    ]
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    

    
    response = completion.choices[0].message.content
    
    messages.append({ "role": "assistant", "content": response })       
    
    print (Fore.CYAN + Style.BRIGHT + "網球教練Frank: " + Style.NORMAL + response)
    
    ai_feedback =  [msg for msg in messages if msg["role"] == "assistant"]
    return ai_feedback 
    
def conclude(INSTRUCTIONS,ai_feedback):
    print ("\nGernerating Conclusion......\n")
    
    messages = [
        { "role": "system", 
          "content": INSTRUCTIONS },
        {"role":"user",
         "content": f"Based on the previous {ai_feedback}, give me a clear and easy-to-understand conclusion about my swing motion in Traditional Chinese, as if you were a coach giving friendly, spoken feedback."} 
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY
    ) 
    
    response = completion.choices[0].message.content
    print ("---CONCLUSION---")
    print ("\n",response)
    
def main():
    target = [f"junior_9_{i}_cleaned" for i in range(3)]
    target_filepaths = [f"./__data__/junior/{t}.json" for t in target]
    
    ai_feedback = []
    
    for t in target_filepaths:
        my_motion = pd.read_json(t)
        coach_motion = pd.read_json(pro_filepath)            
        response = system_message(INSTRUCTIONS, my_motion, coach_motion)
        ai_feedback.append(response)
    
    conclude(INSTRUCTIONS,ai_feedback)

        
    


if __name__ == "__main__":
    main()