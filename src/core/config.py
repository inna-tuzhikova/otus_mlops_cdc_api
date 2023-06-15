from logging import config as logging_config
from enum import Enum

from pydantic import BaseSettings, HttpUrl

from core.logger import LOGGING


logging_config.dictConfig(LOGGING)


class ModelStage(Enum):
    PRODUCTION = 'Production'
    STAGING = 'Staging'


class AppSettings(BaseSettings):
    app_title: str
    host: str
    port: int
    model_name: str
    model_target_stage: ModelStage
    mlflow_s3_endpoint_url: HttpUrl
    mlflow_tracking_uri: HttpUrl

    class Config:
        env_file: str = '.env'


app_settings = AppSettings()
