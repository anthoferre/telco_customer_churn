from typing import List
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session

from app.api.v1.deps import get_db
from app.schemas.item import ItemCreate, ItemUpdate, ItemInDB
from app.services.item_service import item_service
from app.api.v1.auth_deps import get_current_active_user # Si vous voulez protéger les routes

router = APIRouter(
    prefix="/items",
    tags=["Items"],
)

@router.post("/", response_model=ItemInDB)
async def create_item_endpoint(
    item: ItemCreate, 
    db: Session = Depends(get_db),
    # Pour protéger cet endpoint, décommentez la ligne ci-dessous:
    # current_user: Any = Depends(get_current_active_user)
):
    """Crée un nouvel article via le service."""
    db_item = item_service.create_item(db, item=item)
    return {"item": db_item} # Encapsule l'item dans un dictionnaire pour correspondre au test

@router.get("/{item_id}", response_model=ItemInDB)
async def read_item_endpoint(item_id: int, db: Session = Depends(get_db)):
    """Récupère un article par son ID via le service."""
    item = item_service.get_item(db, item_id=item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item non trouvé")
    return {"item": item} # Encapsule l'item dans un dictionnaire pour correspondre au test

@router.get("/", response_model=List[ItemInDB])
async def read_all_items_endpoint(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    """Récupère tous les articles disponibles via le service, avec pagination."""
    items = item_service.get_items(db, skip=skip, limit=limit)
    return items

@router.put("/{item_id}", response_model=ItemInDB)
async def update_item_endpoint(item_id: int, item: ItemUpdate, db: Session = Depends(get_db)):
    """Met à jour un article existant via le service."""
    updated_item = item_service.update_item(db, item_id=item_id, item_in=item)
    if not updated_item:
        raise HTTPException(status_code=404, detail="Item non trouvé pour la mise à jour")
    return {"updated_item": updated_item} # Encapsule l'item dans un dictionnaire pour correspondre au test

@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item_endpoint(item_id: int, db: Session = Depends(get_db)):
    """Supprime un article via le service."""
    deleted_item = item_service.delete_item(db, item_id=item_id)
    if not deleted_item:
        raise HTTPException(status_code=404, detail="Item non trouvé pour la suppression")
    return