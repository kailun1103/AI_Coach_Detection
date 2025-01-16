import os
from dotenv import load_dotenv

import openai as OpenAI
import pandas as pd

import json


class ServiceFeedback:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("GPT_MODEL")
        self.instructions = os.getenv("SYSTEM_PROMPT")
        self.client = OpenAI.Client(api_key= os.getenv("OPENAI_API_KEY"))
        
        self.temperature = 0.5
        self.max_tokens = 500
        self.frequency_penalty = 0
        self.presence_penalty = 0.6

    def system_prompt(self, standard_vector):
        return [
            {"role": "system", "content": self.instructions},
            {"role": "system", "content": f"Here is my tennis swing motion that compared with standard coach: {standard_vector}. Please give me advice on how to improve my swing motion."}
        ]

    def compare_vectors(self, rookie_file_path, standard_file_path, save_path):
        pass

    def generate_feedback(self):
        standard_df = pd.read_json("./__data__/standard/standard.json")
        rookie_df = pd.read_json("./__data__/rookie/rookie.json")
        compared_df = self.compare_vectors(rookie_df, standard_df)

        with open(compared_df, "r") as file:
            compared_data = file.read()

        messages = self.system_prompt(compared_data)

        response = self.client.chat.completions.create(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )

        return response["choices"][0]["message"]["content"]

    def main(self):
        feedback = self.generate_feedback()
        print(feedback)
        
        
if __name__ == "__main__":
    main = ServiceFeedback()
    main.main()

