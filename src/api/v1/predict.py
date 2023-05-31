from logging import getLogger

from fastapi import APIRouter

from schemas.prediction import Prediction, ImageToPredict
from services.prediction import prediction_service


logger = getLogger(__name__)
router = APIRouter()


@router.post(
    '/predict',
    response_model=Prediction,
)
async def predict(image: ImageToPredict):
    """Handler for user image category prediction"""
    prediction = await prediction_service.predict_image(image)
    return prediction
