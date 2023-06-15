import io
import os
from pathlib import Path
from logging import getLogger

from fastapi import UploadFile
import torch
from torchvision import transforms
from PIL import Image
import mlflow

from schemas.prediction import Prediction, ClassLabels
from core.config import app_settings


logger = getLogger(__name__)


class PredictionService:
    def __init__(self, scripted_model_path: Path):
        self._model = ModelWrapper(scripted_model_path)

    async def predict_image(self, image: UploadFile) -> Prediction:
        try:
            contents = await image.read()
            image = Image.open(io.BytesIO(contents))
        except:
            raise ValueError('Cannot read image')
        try:
            image = self._prepare_image(image)
        except:
            raise ValueError('Cannot convert image to RGB')
        try:
            prediction = self._model.predict_image(image)
        except:
            raise ValueError('Unable to analyze image')
        return prediction

    def _prepare_image(self, image: Image) -> Image:
        """Converts non RGB image"""
        if image.getbands() != ('R', 'G', 'B'):
            image = image.convert('RGB')
        return image


class ModelWrapper:
    def __init__(self, scripted_model_path):
        self._model = torch.jit.load(scripted_model_path)
        self._image_to_tensor = transforms.ToTensor()

    def predict_image(self, image: Image) -> Prediction:
        image = self._image_to_tensor(image)
        image = torch.unsqueeze(image, 0)
        probs = self._model(image)[0]
        result = self._decode_result(probs)
        return result

    def _decode_result(self, probs: torch.Tensor) -> Prediction:
        cls_idx = probs.argmax().item()
        cls_prob = probs[cls_idx].item()
        if cls_idx == 0:
            label = ClassLabels.CAT
        elif cls_idx == 1:
            label = ClassLabels.DOG
        else:
            label = None
        return Prediction(
            class_label=label,
            confidence=cls_prob
        )


prediction_service = None


def init_prediction_service() -> None:
    global prediction_service

    scripted_model_path = _fetch_model()
    prediction_service = PredictionService(scripted_model_path)


def _fetch_model() -> Path:
    """Downloads model from mlflow server and saves it locally

    Returns:
        Path - path to fetched model
    """
    mlflow.set_tracking_uri(app_settings.mlflow_tracking_uri)
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = app_settings.mlflow_s3_endpoint_url
    uri = (
        f'models:'
        f'/{app_settings.model_name}'
        f'/{app_settings.model_target_stage.value}'
    )
    logger.info('Fetching model from MLFlow server...')
    save_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=uri,
        dst_path='current_model'
    )
    result = Path(save_dir) / 'scripted_model.pt'
    logger.info('Fetched model from MLFlow server')
    return result


def get_prediction_service() -> PredictionService:
    if prediction_service is None:
        raise RuntimeError(
            'Service is not ready, init it first with init_prediction_service'
        )
    return prediction_service
