from sqlalchemy.orm import Session
from app.models.sql_models import UserDB
from app.schemas.user import UserCreate
from app.core.security import get_password_hash

class CRUDUser:
    def get_user(self, db: Session, user_id: int) -> UserDB | None:
        return db.query(UserDB).filter(UserDB.id == user_id).first()

    def get_user_by_email(self, db: Session, email: str) -> UserDB | None:
        return db.query(UserDB).filter(UserDB.email == email).first()

    def create_user(self, db: Session, user: UserCreate) -> UserDB:
        hashed_password = get_password_hash(user.password)
        db_user = UserDB(
            email=user.email,
            hashed_password=hashed_password,
            full_name=user.full_name,
            is_active=user.is_active,
            is_superuser=user.is_superuser
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

user_crud = CRUDUser()