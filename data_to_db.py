import pandas as pd
from sqlalchemy import create_engine
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Définir les chemins des fichiers et de la base de données ---

# Récupérer le chemin du répertoire parent du script pour une meilleure portabilité
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_FILE = "sql_app.db" 
DB_PATH = os.path.join(BASE_DIR, DB_FILE)
DATABASE_URL = f"sqlite:///{DB_PATH}"

RAW_DATA_FILE = 'raw_data.csv'
RAW_DATA_PATH = os.path.join(BASE_DIR, RAW_DATA_FILE)

# --- 2. Charger les données brutes et effectuer un nettoyage initial ---
def load_and_initial_clean_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données brutes depuis un fichier CSV, effectue un nettoyage initial
    (gestion des valeurs manquantes, suppression des doublons, conversion des types).
    """
    logging.info(f"Démarrage du chargement et nettoyage initial des données depuis '{file_path}'...")
    try:
        # Charger le CSV, en traitant les espaces comme des NaN
        df = pd.read_csv(file_path, na_values=[' ', None])
    except FileNotFoundError:
        logging.error(f"Erreur: Le fichier '{file_path}' n'a pas été trouvé. Veuillez vérifier le chemin.")
        raise # Relance l'exception pour arrêter le script

    logging.info(f"Chargement initial: {df.shape[0]} lignes, {df.shape[1]} colonnes.")

    # Convertir 'TotalCharges' en numérique, en forçant les erreurs à NaN
    # C'est crucial car 'TotalCharges' peut contenir des chaînes vides ou non numériques
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Supprimer les lignes où 'TotalCharges' est devenu NaN après conversion
    # Ou d'autres colonnes essentielles pour le modèle
    initial_rows = df.shape[0]
    df.dropna(how='any', inplace=True) # Supprime les lignes avec n'importe quel NaN restant
    logging.info(f"Après dropna(how='any'): {df.shape[0]} lignes restantes (supprimées: {initial_rows - df.shape[0]}).")

    # Supprimer la colonne 'customerID' si elle n'est pas nécessaire pour le modèle ou la DB
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        logging.info("Colonne 'customerID' supprimée.")
    else:
        logging.warning("La colonne 'customerID' n'a pas été trouvée pour être supprimée.")

    # Normaliser les noms de colonnes pour la cohérence (ex: 'Gender' au lieu de 'gender')
    df.columns = df.columns.str.title()
    logging.info("Noms de colonnes normalisés (titre).")

    logging.info("Données chargées et nettoyées avec succès pour l'ingestion.")
    return df

# --- 3. Fonction principale d'ingestion ---
def ingest_data_to_db():
    """
    Exécute le processus de chargement, nettoyage et insertion des données
    dans la base de données SQLite.
    """
    # 3.1. Charger et nettoyer les données brutes
    df_clean = load_and_initial_clean_data(RAW_DATA_PATH)
    
    if df_clean.empty:
        logging.warning("Aucune donnée nettoyée à insérer dans la base de données.")
        return

    # 3.2. Créer l'engine SQLAlchemy pour la connexion à la base de données
    engine = create_engine(DATABASE_URL)
    logging.info(f"Connexion à la base de données : '{DATABASE_URL}'")

    # 3.3. Spécifier le nom de la table dans la base de données
    TABLE_NAME = "customers"

    # 3.4. Insérer le DataFrame nettoyé dans la base de données SQLite
    try:
        df_clean.to_sql(TABLE_NAME, con=engine, if_exists='replace', index=False)
        logging.info(f"DataFrame nettoyé inséré avec succès dans la table '{TABLE_NAME}' de '{DB_FILE}'.")

        # --- Optionnel : Vérifier le contenu de la table (pour confirmation) ---
        conn = engine.connect()
        df_from_db = pd.read_sql_table(TABLE_NAME, con=conn)
        conn.close()
        logging.info(f"\nPremières lignes des données lues depuis la table '{TABLE_NAME}' :\n{df_from_db.head()}")
        logging.info(f"Total de {df_from_db.shape[0]} lignes dans la table '{TABLE_NAME}'.")

    except Exception as e:
        logging.error(f"Une erreur est survenue lors de l'insertion des données dans la DB : {e}")

# --- Point d'entrée du script ---
if __name__ == "__main__":
    ingest_data_to_db()