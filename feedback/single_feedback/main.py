import os
from openai import OpenAI
from dotenv import load_dotenv
from colorama import Fore, Back, Style
import pandas as pd 
import single_feedback.config as config

load_dotenv()

client = config.client
INSTRUCTIONS = config.INSTRUCTIONS 

TEMPERATURE = config.TEMPERATURE
MAX_TOKENS = config.MAX_TOKENS
FREQUENCY_PENALTY = config.FREQUENCY_PENALTY
PRESENCE_PENALTY = config.PRESENCE_PENALTY
MAX_CONTEXT_QUESTIONS = config.MAX_CONTEXT_QUESTIONS

target = config.target
target_filepath = config.target_filepath


def system_message(INSTRUCTIONS, previous, standard_df):
    messages = [
        { "role": "system", 
          "content": INSTRUCTIONS },
        {"role":"system",
         "content": f"here is my tennis swing motion:{standard_df}. Please base on this give me some advice,let me know how to improve my swing motion."} 
    ]
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    
    response = completion.choices[0].message.content
    messages.append({ "role": "assistant", "content": response })       
    
    print ("\n", Fore.CYAN + Style.BRIGHT + "網球教練Frank: " + Style.NORMAL + response ,"\n")
    previous.append(messages)


def get_response(INSTRUCTIONS, previous_questions_and_answers, new_question, standard_df):
    messages = [ ]
    
    # add the previous questions and answers
    for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": question })
        messages.append({ "role": "assistant", "content": answer })    
        
    # Enter new Question
    messages.append({ "role": "user", "content": new_question })

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content



def get_moderation(question):
    errors = {
        "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
        "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
        "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
        "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
        "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
        "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
    }
    response = client.moderations.create(input=question)
    if response.results[0].flagged:
        result = [
            error
            for category, error in errors.items()
            if response.results[0].categories[category]
        ]
        return result
    return None

def main():
    standard_df = pd.read_json(target_filepath)          
    previous_questions_and_answers = []
    
    system_message(INSTRUCTIONS, previous_questions_and_answers, standard_df)      


if __name__ == "__main__":
    main()