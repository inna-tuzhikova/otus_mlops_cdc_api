import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import app_settings
from api.v1 import predict
from services.prediction import init_prediction_service
from mlflow import MlflowClient

app = FastAPI(
    title=app_settings.app_title,
    docs_url='/api/openapi',
    openapi_url='/api/openapi.json'
)


@app.on_event('startup')
def startup():
    import mlflow.pyfunc

    mlflow.set_tracking_uri('http://51.250.22.177:5000/')
    # mlflow.set_experiment('cdc_test')

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://51.250.22.177:9000'
    model_name = "cdc"
    stage = "Staging"

    # model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
    res = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}/{stage}",
        dst_path='loaded'
    )
    # model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{stage}", dst_path='loaded')
    init_prediction_service(model=None)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    router=predict.router,
    prefix='/api/v1',
    tags=['predict']
)


if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8000,
        reload=True
    )
