from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.core.config import settings
from app.database.connection import create_db_and_tables, SessionLocal
from app.crud.crud_user import user_crud
from app.schemas.user import UserCreate
from app.core.ml_models import load_churn_model

# --- Initialisation de l'application FastAPI ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    version="0.0.1"
)

# --- Configuration CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Crée les tables de la base de données et l'utilisateur initial au démarrage ---
@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    db = SessionLocal()
    try:
        # Crée un superuser initial si aucun superuser n'existe.
        # Vous pouvez modifier l'email/mot de passe par défaut ici.
        initial_user_email = "admin@example.com"
        initial_user_password = "password" 

        user = user_crud.get_user_by_email(db, email=initial_user_email)
        if not user:
            user_in = UserCreate(
                email=initial_user_email,
                password=initial_user_password,
                is_superuser=True,
                full_name="Admin User"
            )
            user_crud.create_user(db, user=user_in)
            print(f"Superuser '{initial_user_email}' créé avec succès.")
        else:
            print(f"Superuser '{initial_user_email}' existe déjà.")
    finally:
        db.close()
    
    # Charge le modèle de Machine Learning
    load_churn_model()

# --- Gestionnaire d'exception personnalisé ---
class CustomException(HTTPException):
    def __init__(self, name: str):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Oups, une erreur personnalisée est survenue : {name}")
        self.name = name

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"Désolé, l'action pour '{exc.name}' a échoué. Cause: {exc.detail}"},
    )

# --- Inclure le routeur principal de votre API ---
app.include_router(api_router, prefix=settings.API_V1_STR)

# --- Routes de base ---
@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API ! (Version principale)"}

@app.get("/trigger-custom-error/{name}")
async def trigger_custom_error(name: str):
    if name == "test_fail":
        raise CustomException(name=name)
    return {"message": f"L'opération pour '{name}' a réussi."}