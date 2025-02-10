import os
from openai import OpenAI

# ------------------- KEY VARIABLE -------------------
client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))

INSTRUCTIONS = """
Act as a strict tennis coach specializing in observing swing motion vectors and providing detailed feedback for tennis beginners.
Always use a friendly and patient tone when responding.
All responses must be presented in paragraph format and should never use bullet points.
All responses must be in Traditional Chinese, and no bullet points should be used.
Ensure that the feedback is output only in a structured JSON format, similar to the following {example}:

{example}
{
"problem_frame": "20-30",
"suggestion": "此軌跡範圍有問題，速度太快，揮拍過低"
}       
"""
# 這邊可能要調整-----------------------------------------------------------------------------------------------------------------
PROMPT_1 = """
in the next convesation i will give you a json file, 
This JSON file contains multiple frames of vectors describing the trajectory of a tennis swing. 
And the data schema includes frames, 3D spatial coordinates of right wrist, right elbow, right shoulder, 
and a boolean indicating whether the tennis ball was hit
"""


          
TEMPERATURE = 0.5
MAX_TOKENS = 300
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
MAX_CONTEXT_QUESTIONS = 10
TOP_P =1

# ------------------- Target file -------------------

pro = "pro_cleaned"
pro_filepath = f"./__data__/pro/{pro}.json"

