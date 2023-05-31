import random

from schemas.prediction import ImageToPredict, Prediction, ClassLabels


class PredictionService:
    def __init__(self):
        pass

    async def predict_image(self, image: ImageToPredict) -> Prediction:
        return Prediction(
            class_label=random.choice(list(ClassLabels)),
            confidence=.5 + .5*random.random()
        )


prediction_service = PredictionService()
