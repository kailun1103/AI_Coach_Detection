import openai
from schema.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_chat_response(user_input):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=user_input,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"