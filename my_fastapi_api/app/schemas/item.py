from pydantic import BaseModel, Field
from typing import Union, List, Optional

class ItemBase(BaseModel):
    name: str = Field(min_length=1)
    description: Union[str, None] = None
    price: float = Field(gt=0)
    tax: Union[float, None] = None
    is_offered: bool = False

class ItemCreate(ItemBase):
    pass

class ItemUpdate(ItemBase):
    pass

class ItemInDB(ItemBase):
    id: int

    class Config:
        from_attributes = True