from logging import config as logging_config

from pydantic import BaseSettings

from core.logger import LOGGING


logging_config.dictConfig(LOGGING)


class AppSettings(BaseSettings):
    app_title: str
    host: str
    port: int

    class Config:
        env_file: str = '.env'


app_settings = AppSettings()
