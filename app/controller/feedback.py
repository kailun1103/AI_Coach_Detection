from fastapi import HTTPException
from typing import Dict, Any

from ..service.feedback import service_feedback


class FeedbackController:
    def init(self):
        self.service = FeedbackService()