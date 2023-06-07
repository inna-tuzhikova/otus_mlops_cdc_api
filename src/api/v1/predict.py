from logging import getLogger

from fastapi import APIRouter, UploadFile, Depends, HTTPException, status

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
    try:
        prediction = await prediction_service.predict_image(image)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    return prediction
