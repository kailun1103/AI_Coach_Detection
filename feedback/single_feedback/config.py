import os
from openai import OpenAI

# ------------------- KEY VARIABLE -------------------
client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))

INSTRUCTIONS = """
Act as a strict tennis mentor , expertising in observe swing motion vector and provide feedback for tennis beginners.  
Always Use a friendly and patient tone.
Always respond in paragraph format, never using bullet points.  
Always answer in Traditional Chinese.
Never answer in bullet point  
If a question is unclear, kindly remind the user to provide more details.                
"""


          
TEMPERATURE = 0.5
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
MAX_CONTEXT_QUESTIONS = 10


# ------------------- Target file -------------------
target = "junior_9_1_cleaned"
target_filepath = f"./__data__/junior/{target}.json"

pro = "pro_cleaned"
pro_filepath = f"./__data__/pro/{pro}.json"

