from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ....controller.feedback import 
from ....schema.feedback import FeedbackResponse, FeedbackRequest


router = APIRouter()

@router.post(
    "/generate/response",  
    response_model=  FeedbackResponse
)
async def generate_response(
    request: FeedbackRequest
    controller: FeedbackController = Depends()
) -> FeedbackResponse:
    return await controller.




@router.get("/healthcheck")
async def health_check():
    return {"status": "healthy"}