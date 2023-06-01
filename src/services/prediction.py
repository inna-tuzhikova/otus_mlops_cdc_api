import random

from fastapi import UploadFile
import aiofiles

from schemas.prediction import Prediction, ClassLabels


class PredictionService:
    def __init__(self):
        pass

    async def predict_image(self, image: UploadFile) -> Prediction:
        # tmp_path = f'/home/inna/cdc_input'
        # async with aiofiles.open(tmp_path, 'wb') as out_file:
        #     while content := await image.read(1024):  # async read chunk
        #         await out_file.write(content)  # async write chunk
        return Prediction(
            class_label=random.choice(list(ClassLabels)),
            confidence=.5 + .5*random.random()
        )


prediction_service = PredictionService()
