from fastapi import FastAPI
from pydantic import BaseModel, Field, validator


class FeedbackResponse(BaseModel):
    text = str
    usage = [str,int]

class FeedbackRequest(BaseModel):
    user_input: str = Field(..., title="User Input", description="User input for the chatbot", example="I am feeling sad today")
    role: messagerole = Field(..., title="Role", description="Role of the user", example="user")
    
    
    
class messagerole (str, Enum):
    USER = "user"
    ASSISTANT = "assistant"