import os
from openai import OpenAI
from dotenv import load_dotenv
from colorama import Fore, Back, Style
import pandas as pd 

load_dotenv()

# ------------------- KEY VARIABLE -------------------
client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))

INSTRUCTIONS = """
Act as a tennis coach named Frank, providing guidance and answering questions specifically for tennis beginners.  
Use a friendly and patient tone, similar to that of a caring and attentive coach, and respond in paragraph format.  
Always answer in Traditional Chinese.  
If a question is unclear, kindly remind the user to provide more details.  
If you encounter a topic you don’t know or a question unrelated to tennis, honestly say you don’t know.  
              """
              
TEMPERATURE = 0.5
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
MAX_CONTEXT_QUESTIONS = 10

standard_file = "sim_1"
standard_filepath = f"./feedback/data/standard_player/{standard_file}.json"


def get_response(INSTRUCTIONS, previous_questions_and_answers, new_question, standard_df):
    messages = [
        { "role": "system", 
          "content": INSTRUCTIONS },
        {"role":"system",
         "content": f"Here is body vector data for a standard tennis player's swing motion. Please remember this data. standard tennis player's swing motion:{standard_df}"}
        
        
        
    ]
    # add the previous questions and answers
    for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": question })
        messages.append({ "role": "assistant", "content": answer })    
        
    # Enternew Question
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
    standard_df = pd.read_json(standard_filepath)          

    previous_questions_and_answers = []
    while True:
        # ask the user for their question
        new_question = input(Fore.GREEN + Style.BRIGHT + "想問些什麼？: " + Style.RESET_ALL)
        
        if new_question.lower() == "exit":
            print("\n對話已結束。\n")
            break
        
        # check the question is safe
        errors = get_moderation(new_question)
        if errors:
            print(
                Fore.RED
                + Style.BRIGHT
                + "Sorry, you're question didn't pass the moderation check:"
            )
            for error in errors:
                print(error)
            print(Style.RESET_ALL)
            continue
        
        print ("\n(Thinking.........)\n")
       
        response = get_response(INSTRUCTIONS, previous_questions_and_answers, new_question, standard_df)

        # add the new question and answer to the list of previous questions and answers
        previous_questions_and_answers.append((new_question, response))

        # print the response
        print("\n",Fore.CYAN + Style.BRIGHT + "網球教練Frank: " + Style.NORMAL + response,"\n")


if __name__ == "__main__":
    main()