from fastapi import APIRouter, Depends, HTTPException, status
from .endpoint import feedback 



router = APIRouter(prefix="/ai_tennis_coach")

router.include_router(feedback.router, prefix="/feedback")   