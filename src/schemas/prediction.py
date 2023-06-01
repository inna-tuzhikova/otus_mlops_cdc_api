from pydantic.main import BaseModel
from pydantic.types import Enum


class ClassLabels(Enum):
    CAT = 'cat'
    DOG = 'dog'


class Prediction(BaseModel):
    confidence: float
    class_label: ClassLabels
