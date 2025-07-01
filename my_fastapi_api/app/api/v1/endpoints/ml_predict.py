import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
# from sqlalchemy.orm import Session # Décommenter si besoin d'enregistrer des prédictions

from app.schemas.ml_prediction import ChurnPredictionInput, ChurnPredictionOutput
from app.core.ml_models import churn_model # , load_churn_model si vous voulez le recharger ici
# from app.api.v1.auth_deps import get_current_active_user # Si vous voulez protéger cet endpoint

router = APIRouter()

@router.post("/predict_churn", response_model=ChurnPredictionOutput, tags=["Machine Learning"])
async def predict_churn(
    input_data: ChurnPredictionInput,
    # current_user: Any = Depends(get_current_active_user) # Exemple de protection
):
    """
    Endpoint pour prédire le churn client à l'aide du modèle ML.
    Prend en entrée les caractéristiques du client et renvoie une prédiction.
    """
    if churn_model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Modèle de ML non chargé. Veuillez contacter l'administrateur.",
        )

    features = np.array([[input_data.feature1, input_data.feature2]])

    prediction = churn_model.predict(features)[0]
    prediction_proba = churn_model.predict_proba(features)[0]

    confidence = float(prediction_proba[prediction])
    if prediction == 1:
        message = "Le modèle prédit un risque de churn élevé."
    else:
        message = "Le modèle prédit un faible risque de churn."

    return ChurnPredictionOutput(
        prediction=int(prediction),
        confidence=confidence,
        message=message
    )