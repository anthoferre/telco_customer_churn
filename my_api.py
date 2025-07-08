from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
import os
import joblib
import contextlib
import pandas as pd
import logging
from typing import Optional, List
from sqlmodel import create_engine, SQLModel, Field as SQLField, Session, select

# --- Configuration du logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- Initialisation de l'application FastAPI ---
api = FastAPI(title='API de prédiction de Churn et gestion client', version='1.0.0')

# --- Chemin de la pipeline à charger ---
churn_model = None
MODEL_PATH = "best_overall_churn_model.pkl" # Assurez-vous que ce fichier existe

# --- Chemin de la database à charger ---
DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(DATABASE_URL, echo=False) # echo=False pour ne pas spammer les logs avec les requêtes SQL

# --- Modèle SQLModel pour la table des clients ---
class Customer(SQLModel, table=True):
    """
    Modèle de base de données pour un client, correspondant aux champs d'entrée.
    """
    id: Optional[int] = SQLField(default=None, primary_key=True)
    Gender: str
    Seniorcitizen: int
    Partner: str
    Dependents: str
    Tenure: int
    Phoneservice: str
    Multiplelines: str
    Internetservice: str
    Onlinesecurity: str
    Onlinebackup: str
    Deviceprotection: str
    Techsupport: str
    Streamingtv: str
    Streamingmovies: str
    Contract: str
    Paperlessbilling: str
    Paymentmethod: str
    Monthlycharges: float
    Totalcharges: float

    class Config:
        arbitrary_types_allowed = True


# --- Fonctions pour la gestion de la base de données ---
def create_db_and_tables():
    """
    Crée les tables de la base de données si elles n'existent pas.
    """
    SQLModel.metadata.create_all(engine)
    logger.info("Tables de la base de données créées ou déjà existantes.")

def get_session():
    """
    Dépendance pour obtenir une session de base de données.
    Utilisée avec `Depends` dans les routes.
    """
    with Session(engine) as session:
        yield session

# --- Chargement du modèle au démarrage de l'application ---
@contextlib.asynccontextmanager
async def lifespan(api: FastAPI):
    """
    Gère les événements de démarrage et d'arrêt de l'application FastAPI.
    """
    global churn_model

    logger.info("Démarrage de l'application API.")

    # Vérifie si le fichier du modèle existe avant de tenter de le charger
    if not os.path.exists(MODEL_PATH):
        logger.error(f'Erreur : Fichier modèle introuvable à {MODEL_PATH}')
        raise FileNotFoundError(f'Modèle de ML non trouvé. Veuillez placer {MODEL_PATH} dans le répertoire de l\'API.')
    
    try:
        # Charge le pipeline complet (qui inclut le préprocesseur)
        churn_model = joblib.load(MODEL_PATH)
        logger.info(f"Modèle de churn (pipeline complet) chargé avec succès depuis {MODEL_PATH}")
    except Exception as e:
        logger.exception(f'Erreur lors du chargement du modèle : {e}')
        raise
    
    # Création des tables de la base de données
    create_db_and_tables()

    yield

    # --- Code exécuté lorsque l'application s'arrête ---
    logger.info("Arrêt de l'application API.")

# Associe la fonction lifespan à l'API
api.router.lifespan_context = lifespan

# --- Schémas Pydantic pour les données d'entrée/sortie ---

class ChurnPredictionInput(BaseModel):
    """
    Définit le schéma des données d'entrées pour la prédiction de churn.
    Ces champs correspondent aux caractéristiques du client.
    """
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

# --- Routes de l'API ---

@api.post("/predict_churn", response_model=ChurnPredictionOutput, summary="Prédire le désabonnement client")
async def predict_churn(input_data: ChurnPredictionInput):
    """
    Endpoint pour la prédiction de désabonnement des clients.
    
    **Entrées** :
    - Caractéristiques du client (voir le schéma `ChurnPredictionInput`).
    
    **Sorties** :
    - La probabilité que le client se désabonne.
    - La prédiction binaire ('Yes' ou 'No') basée sur cette probabilité.
    """
    global churn_model

    if churn_model is None:
        logger.error("Le modèle de prédiction n'a pas été chargé lors du démarrage.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Le modèle de prédiction n'a pas été chargé.")

    # Convertit les données reçues en entrée dans l'API (pydantic base model) en un DataFrame pandas.
    # Le pipeline chargé (churn_model) gérera automatiquement le pré-traitement.
    input_df_raw = pd.DataFrame([input_data.dict()])
    logger.debug(f"Données brutes reçues pour prédiction: {input_df_raw.to_dict()}")

    try:
        # Effectue la prédiction en utilisant le pipeline complet
        # Le pipeline inclut le préprocesseur et le sélecteur de caractéristiques.
        # Plus besoin d'appeler `preprocess_for_prediction` manuellement.
        churn_probability = float(churn_model.predict_proba(input_df_raw)[0][1])
        
        # NOTE : La variable CHURN_THRESHOLD n'est pas définie dans l'API.
        # Vous devriez soit la définir ici, soit la rendre configurable.
        # Pour l'instant, j'utilise 0.5 comme valeur par défaut si non définie.
        PREDICTION_THRESHOLD = 0.5 
        churn_label = "Yes" if churn_probability >= PREDICTION_THRESHOLD else "No" 
        logger.info(f"Prédiction pour le client: Probabilité={churn_probability:.2f}, Label={churn_label}")
    
    except Exception as e:
        logger.exception(f"Erreur lors de la prédiction du modèle : {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erreur lors de la prédiction : {e}. Vérifiez les données d'entrées.")
    
    return ChurnPredictionOutput(
        churn_probability=churn_probability,
        churn_label=churn_label
    )

---
### **Routes pour la gestion des clients (CRUD)**

```python
@api.post("/customers/", response_model=Customer, status_code=status.HTTP_201_CREATED, summary="Ajouter un nouveau client")
async def create_customer(customer: ChurnPredictionInput, session: Session = Depends(get_session)):
    """
    Crée un nouvel enregistrement client dans la base de données.
    """
    try:
        db_customer = Customer(**customer.dict())
        session.add(db_customer)
        session.commit()
        session.refresh(db_customer)
        logger.info(f"Client ajouté avec succès : ID={db_customer.id}")
        return db_customer
    except Exception as e:
        logger.exception(f"Erreur lors de l'ajout du client : {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Impossible d'ajouter le client : {e}")

@api.get("/customers/", response_model=List[Customer], summary="Récupérer tous les clients")
async def read_customers(session: Session = Depends(get_session)):
    """
    Récupère la liste de tous les clients enregistrés dans la base de données.
    """
    try:
        customers = session.exec(select(Customer)).all()
        logger.info(f"Récupération de {len(customers)} clients.")
        return customers
    except Exception as e:
        logger.exception(f"Erreur lors de la récupération des clients : {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Impossible de récupérer les clients : {e}")

@api.get("/customers/{customer_id}", response_model=Customer, summary="Récupérer un client par ID")
async def read_customer(customer_id: int, session: Session = Depends(get_session)):
    """
    Récupère un client spécifique par son ID.
    """
    customer = session.get(Customer, customer_id)
    if not customer:
        logger.warning(f"Client introuvable avec l'ID : {customer_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client non trouvé")
    logger.info(f"Client récupéré avec l'ID : {customer_id}")
    return customer

@api.put("/customers/{customer_id}", response_model=Customer, summary="Mettre à jour un client existant")
async def update_customer(customer_id: int, customer_update: ChurnPredictionInput, session: Session = Depends(get_session)):
    """
    Met à jour les informations d'un client existant.
    """
    customer = session.get(Customer, customer_id)
    if not customer:
        logger.warning(f"Tentative de mise à jour d'un client introuvable avec l'ID : {customer_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client non trouvé")
    
    try:
        # Met à jour l'objet customer avec les données de customer_update
        for field, value in customer_update.dict(exclude_unset=True).items():
            setattr(customer, field, value)
        
        session.add(customer)
        session.commit()
        session.refresh(customer)
        logger.info(f"Client mis à jour avec succès : ID={customer_id}")
        return customer
    except Exception as e:
        logger.exception(f"Erreur lors de la mise à jour du client {customer_id} : {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Impossible de mettre à jour le client : {e}")

@api.delete("/customers/{customer_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Supprimer un client")
async def delete_customer(customer_id: int, session: Session = Depends(get_session)):
    """
    Supprime un client de la base de données.
    """
    customer = session.get(Customer, customer_id)
    if not customer:
        logger.warning(f"Tentative de suppression d'un client introuvable avec l'ID : {customer_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client non trouvé")
    
    try:
        session.delete(customer)
        session.commit()
        logger.info(f"Client supprimé avec succès : ID={customer_id}")
        return {}
    except Exception as e:
        logger.exception(f"Erreur lors de la suppression du client {customer_id} : {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Impossible de supprimer le client : {e}")