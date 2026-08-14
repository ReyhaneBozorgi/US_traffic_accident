from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    description='US Accident API',
    version='1.0.0'
)


app.include_router(router)