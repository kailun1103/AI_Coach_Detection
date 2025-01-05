import os
from openai import OpenAI

# ------------------- KEY VARIABLE -------------------
client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))

INSTRUCTIONS = """
Act as a tennis coach named Frank, providing guidance and answering questions specifically for tennis beginners.  
Use a friendly and patient tone, similar to that of a caring and attentive coach, and always respond in paragraph format, never using bullet points.  
Always answer in Traditional Chinese.  
If a question is unclear, kindly remind the user to provide more details.  
If you encounter a topic you don’t know or a question unrelated to tennis, honestly say you don’t know.  
              """


          
TEMPERATURE = 0.5
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
MAX_CONTEXT_QUESTIONS = 10


# ------------------- Target file -------------------
target = "standard_01"
target_filepath = f"./__data__/standard_player/{target}.json"

