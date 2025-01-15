import openai
import os
import pandas as pd

class ServiceFeedback:
    def __init__(self):
        self.client = openai.chat.completions()
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.instructions = "Act as a tennis coach"
        self.temperature = 0.5
        self.max_tokens = 500
        self.frequency_penalty = 0
        self.presence_penalty = 0.6
        self.model = "gpt-4o"

    def system_prompt(self, standard_vector):
        return [
            {"role": "system", "content": self.instructions},
            {"role": "system", "content": f"Here is my tennis swing motion that compared with standard coach: {standard_vector}. Please give me advice on how to improve my swing motion."}
        ]

    def read_standard_df(self):
        standard_file = "./__data__/standard/standard_df.json"
        return pd.read_json(standard_file)

    def read_rookie_df(self):
        rookie_file = "./__data__/rookie/rookie_df.json"
        return pd.read_json(rookie_file)

    def compare_vectors(self):
        standard_df = self.read_standard_df()
        rookie_df = self.read_rookie_df()
        difference_df = rookie_df - standard_df
        difference_file = "./__data__/compared/compared_df.json"
        difference_df.to_json(difference_file)
        return difference_file

    def generate_feedback(self, user_input):
        standard_df = self.read_standard_df()
        compared_file = self.compare_vectors()

        with open(compared_file, "r") as file:
            compared_data = file.read()

        messages = self.system_prompt(compared_data)
        messages.append({"role": "user", "content": user_input})

        response = openai.ChatCompletion.create(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )

        return response["choices"][0]["message"]["content"]

