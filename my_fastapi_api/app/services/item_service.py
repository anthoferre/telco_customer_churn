from sqlalchemy.orm import Session
from typing import List, Optional

from app.models.sql_models import ItemDB
from app.schemas.item import ItemCreate, ItemUpdate

class ItemService:
    def get_item(self, db: Session, item_id: int) -> Optional[ItemDB]:
        return db.query(ItemDB).filter(ItemDB.id == item_id).first()

    def get_items(self, db: Session, skip: int = 0, limit: int = 10) -> List[ItemDB]:
        return db.query(ItemDB).offset(skip).limit(limit).all()

    def create_item(self, db: Session, item: ItemCreate) -> ItemDB:
        db_item = ItemDB(
            name=item.name,
            description=item.description,
            price=item.price,
            tax=item.tax,
            is_offered=item.is_offered
        )
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        return db_item

    def update_item(self, db: Session, item_id: int, item_in: ItemUpdate) -> Optional[ItemDB]:
        db_item = db.query(ItemDB).filter(ItemDB.id == item_id).first()
        if not db_item:
            return None

        for key, value in item_in.model_dump(exclude_unset=True).items():
            setattr(db_item, key, value)

        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        return db_item

    def delete_item(self, db: Session, item_id: int) -> Optional[ItemDB]:
        db_item = db.query(ItemDB).filter(ItemDB.id == item_id).first()
        if not db_item:
            return None
        
        db.delete(db_item)
        db.commit()
        return db_item

item_service = ItemService()