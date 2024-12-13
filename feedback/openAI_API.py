from openai import OpenAI

client = OpenAI(
)


# 初始化對話歷史
conversation_history = [
    {"role": "system", "content": "你是一個有記憶的聊天機器人，可以幫助解答用戶問題。且永遠會用繁體中文回答我"}
]

def chatbot_with_memory(user_input):
    global conversation_history

    # 將用戶輸入加入對話歷史
    conversation_history.append({"role": "user", "content": user_input})
    
    # 呼叫 OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",  # 修正為正確的模型名稱
        messages=conversation_history,
        temperature=0.7
    )
    
    # 提取回應內容
    assistant_reply = response.choices[0].message.content  # 使用正確的屬性訪問方式
    
    # 將機器人回應加入對話歷史
    conversation_history.append({"role": "assistant", "content": assistant_reply})
    
    return assistant_reply

# 測試對話
if __name__ == "__main__":
    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "quit"]:
            break
        print("機器人：" + chatbot_with_memory(user_input))
