import uvicorn
from fastapi import FastAPI, Request, Response

from core.config import app_settings
from api.v1 import predict


app = FastAPI(
    title=app_settings.app_title,
    docs_url='/api/openapi',
    openapi_url='/api/openapi.json'
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
