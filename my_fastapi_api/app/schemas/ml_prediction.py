from pydantic import BaseModel, Field
from typing import List

class ChurnPredictionInput(BaseModel):
    feature1: float = Field(..., description="Première caractéristique numérique")
    feature2: float = Field(..., description="Deuxième caractéristique numérique")

    class Config:
        schema_extra = {
            "example": {
                "feature1": 5.5,
                "feature2": 8.2
            }
        }

class ChurnPredictionOutput(BaseModel):
    prediction: int = Field(..., description="Prédiction du modèle (ex: 0 pour non-churn, 1 pour churn)")
    confidence: float = Field(..., description="Confiance de la prédiction", ge=0.0, le=1.0)
    message: str = Field(..., description="Message explicatif de la prédiction")