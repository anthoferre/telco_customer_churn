from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import HttpUrl
from typing import List, Union

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'
    )

    PROJECT_NAME: str = "API de Gestion du Churn Telco"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "votre-super-cle-secrete-ici-ceci-doit-etre-une-chaine-longue-et-aleatoire"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    DATABASE_URL: str
    SQL_ALCHEMY_DATABASE_URL: str

    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost", "http://localhost:8080"]

settings = Settings()