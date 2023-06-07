import io
from pathlib import Path

from fastapi import UploadFile
import torch
from torchvision import transforms
from PIL import Image

from schemas.prediction import Prediction, ClassLabels


class PredictionService:
    def __init__(self):
        self._model = ModelWrapper()

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

    def _prepare_image(self, image: Image):
        """Converts non RGB image"""
        if image.getbands() != ('R', 'G', 'B'):
            image = image.convert('RGB')
        return image


class ModelWrapper:
    def __init__(self):
        self._model = torch.jit.load(self._model_path())
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

    def _model_path(self):
        return Path(__file__).parent / 'scripted_predictor.pt'


prediction_service = None


def init_prediction_service():
    global prediction_service
    prediction_service = PredictionService()


def get_prediction_service() -> PredictionService:
    if prediction_service is None:
        raise RuntimeError(
            'Service is not ready, init it first with init_prediction_service'
        )
    return prediction_service
