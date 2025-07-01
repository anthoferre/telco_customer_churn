from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.api.v1.deps import get_db
from app.api.v1.auth_deps import get_current_active_user
from app.schemas.token import Token
from app.schemas.user import User
from app.core.security import verify_password, create_access_token
from app.core.config import settings
from app.crud.crud_user import user_crud

router = APIRouter()

@router.post("/token", response_model=Token)
async def login_access_token(
    db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """OAuth2 compatible token login, obtient un token d'accès pour les futures requêtes."""
    user = user_crud.get_user_by_email(db, email=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email ou mot de passe incorrect.",
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        ),
        "token_type": "bearer",
    }

@router.get("/me/", response_model=User)
async def read_users_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Récupère les informations de l'utilisateur courant."""
    return current_user

@router.get("/me/items/", response_model=list[str])
async def read_own_items(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Récupère les articles de l'utilisateur courant (exemple)."""
    return ["Item 1 de " + current_user.email, "Item 2 de " + current_user.email]