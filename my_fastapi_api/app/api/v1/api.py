from fastapi import APIRouter
from app.api.v1.endpoints import items, login, ml_predict

api_router = APIRouter()
api_router.include_router(items.router)
api_router.include_router(login.router)
api_router.include_router(ml_predict.router)