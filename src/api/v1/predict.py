from logging import getLogger

from fastapi import APIRouter, UploadFile, Depends

from schemas.prediction import Prediction
from services.prediction import PredictionService, get_prediction_service

logger = getLogger(__name__)
router = APIRouter()


@router.post(
    '/predict',
    response_model=Prediction,
)
async def predict(
    image: UploadFile,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """Handler for user image category prediction"""
    prediction = await prediction_service.predict_image(image)
    return prediction
