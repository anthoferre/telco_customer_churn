import pickle
import os
from typing import Any

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dummy_churn_model.pkl")

churn_model: Any = None

def load_churn_model():
    global churn_model
    if churn_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modèle ML introuvable à : {MODEL_PATH}. Veuillez exécuter 'python scripts/train_dummy_model.py' d'abord.")
        with open(MODEL_PATH, 'rb') as f:
            churn_model = pickle.load(f)
        print(f"Modèle de churn chargé depuis : {MODEL_PATH}")
    return churn_model