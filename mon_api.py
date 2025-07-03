from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import joblib
import contextlib
import pandas as pd
from sqlmodel import create_engine, SQLModel

# --- 1. Initialisation de l'application FastAPI ---

api = FastAPI(title = 'API de prédiction de Churn',
              version='1.0.0')

# --- 2. Chemin de la pipeline à charger ---
churn_model = None
MODEL_PATH = "churn_model.pkl"

# --- 3. Chemin de la database à charger
DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(DATABASE_URL, echo = True)

# 4. Fonction pour créer les tables
def create_db_and_tables():
    """
    Crée les tables de la base de données si elles n'existent pas.
    """
    SQLModel.metadata.create_all(engine)

# --- 4. Chargement du modèle au démarrage de l'application ---
@contextlib.asynccontextmanager
async def lifespan(api: FastAPI):
    """
    Gère les événements de démarrage et d'arrêt de l'application FastAPI.
    Le code avant 'yield' s'exécute au démarrage (startup).
    Le code après 'yield' s'exécute à l'arrêt (shutdown).
    """
    # nous modifions la variable globale 'churn_model'
    global churn_model

    # vérifie si le fichier du modèle existe avant de tenter de le charger
    if not os.path.exists(MODEL_PATH):
        print(f'Erreur : Fichier modèle introuvable à {MODEL_PATH}')
        # lève une erreur pour empêcher l'application de démarrer si le fichier est manquant
        raise FileNotFoundError(f'''Modèle de ML non trouvé. Veuillez placer {MODEL_PATH} dans le répertoire de l'API''')
    
    try:
        # charge le fichier 
        churn_model = joblib.load(MODEL_PATH)
        print(f"Modèle de churn chargé avec succès depuis {MODEL_PATH}")
    except Exception as e:
        print(f'Erreur lors du chargement du modèle {e}')
        # lève une erreur pour empêcher l'application de démarrer si le chargement du fichier ne se fait pas
        raise
    
    # création des tables de la base de données
    create_db_and_tables()

    yield

    # --- Code exécuté lorsque l'application s'arrête
    print('''Arrêt de l'application''')

# associe la fonction lifespan à l'API
api.router.lifespan_context = lifespan

class ChurnPredictionInput(BaseModel):
    """
    Définit le schéma des données d'entrées pour la prédiction de churn.
    Ces champs correspondent aux caractéristiques du client."""

    Gender : str = Field(..., example='Male', description="Genre du client (Male/Female)", pattern="^(Male|Female)$")
    Seniorcitizen : int = Field(..., example=1, description="Indique si le client est un sénior (1) ou non (0)", ge=0, le=1)
    Partner : str = Field(..., example='Yes', description="Indique si le client a un partenaire (Yes/No)", pattern="^(Yes|No)$")
    Dependents : str = Field(..., example='Yes', description="Indique si le client est dépendant (Yes/No)", pattern="^(Yes|No)$")
    Tenure : int = Field(..., example=10, description="Nombre de mois depuis que le client est abonné", ge=1)
    Phoneservice : str = Field(..., example='Yes', description="Indique si le client dispose d'un service téléphonique (Yes/No)", pattern="^(Yes|No)$")
    Multiplelines : str = Field(..., example='Yes', description="Indique si le client a plusieurs lignes téléphoniques (Yes/No/No phone service)", pattern="^(Yes|No|No phone service)$")
    Internetservice : str = Field(..., example='Yes', description="Indique si le client dispose d'un service internet (DSL/Fiber optic/No)", pattern="^(DSL|Fiber optic|No)$")
    Onlinesecurity : str = Field(..., example='Yes', description="Indique si le client dispose d'une service sécurité en ligne (Yes/No/No internet service)", pattern="^(Yes|No|No internet service)$")
    Onlinebackup : str = Field(..., example='Yes', description="Indique si le client dispose d'un service de sauvegarde en ligne (Yes/No/No internet service)", pattern="^(Yes|No|No internet service)$")
    Deviceprotection : str = Field(..., example='Yes', description="Indique si le client dispose d'un service de protection des appareils (Yes/No/No internet service)", pattern="^(Yes|No|No internet service)$")
    Techsupport : str = Field(..., example='Yes', description="Indique si le client dispose d'un service de support technique (Yes/No/No internet service)", pattern="^(Yes|No|No internet service)$")
    Streamingtv : str = Field(..., example='Yes', description="Indique si le client dispose d'un service de streaming télé (Yes/No/No internet service)", pattern="^(Yes|No|No internet service)$")
    Streamingmovies : str = Field(..., example='Yes', description="Indique si le client dispose d'un service de films en streaming  (Yes/No/No internet service)", pattern="^(Yes|No|No internet service)$")
    Contract : str = Field(..., example='Yes', description="Indique le type d'abonnement (Month-to-month/One year/Two year)", pattern="^(Month-to-month|One year|Two year)$")
    Paperlessbilling : str = Field(..., example='Yes', description="Indique si le client a ses factures papiers ou non (Yes/No)", pattern="^(Yes|No)$")
    Paymentmethod : str = Field(..., example='Yes', description="Moyen de paiement(Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))", pattern="^(Electronic check|Mailed check|Bank transfer (automatic)|Credit card (automatic))$")
    Monthlycharges : float = Field(..., example='80.5', description="Charges mensuelles", ge=0)
    Totalcharges : float = Field(..., example='400.5', description="Charges totales", ge=0)


class ChurnPredictionOutput(BaseModel):
    """
    Définit le schéma des données attendus en sortie après la prédiction du churn.
    """
    churn_probability : float = Field(..., description="Probabilité de désabonnement du client (valeur entre 0 et 1)", ge=0, le=1)
    churn_label : str = Field(..., description="Prédiction du désabonnement : 'Yes' si la probabilité >= 0.5 sinon 'No'")


# --- Route de prédiction ---
@api.post("/predict_churn", response_model=ChurnPredictionOutput, summary="Prédire le désabonnement client")
async def predict_churn(input_data: ChurnPredictionInput):
    """
    Endpoint pour la prédiction de désabonnement des clients.
    
    Entrées :
    - caractéristiques du client
    
    Sorties :
    - La probabilité que le client se désabonne.
    - La prédiction basée sur cette probabilité.
    """

    global churn_model

    if churn_model is None:
        raise HTTPException(status_code=500, detail = "Le modèle de prédiction n'a pas été chargé.")

    # convertit les données reçues en entrée dans l'API (pydantic base model) en un dataframe pandas
    input_df = pd.DataFrame([input_data.dict()])

    try:
        churn_probability = float(churn_model.predict_proba(input_df)[0][1])

        churn_label = "Yes" if churn_probability >= 0.5 else "No"
    
    except Exception as e:
        print(f"Erreur lors de la prédiction du modèle {e}")
        raise HTTPException(status_code=500, detail = f"Erreur lors de la prédiction du modèle {e}. Vérifier que les données d'entrées sont correctes")
    
    return ChurnPredictionOutput(
        churn_probability=churn_probability,
        churn_label=churn_label
    )
