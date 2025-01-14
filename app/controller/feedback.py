@app.get("/chat")
async def chat(input: str = Query(..., description="User input for chatbot")):
    try:
        response = get_chat_response(input)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))