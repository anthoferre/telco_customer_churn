import pandas as pd
from sqlalchemy import create_engine
import os
import joblib

# --- 1. Définir le chemin de votre base de données SQLite ---
# Il est préférable de placer le fichier .db à la racine du projet ou dans un dossier 'data'
DB_FILE = "sql_app.db" # Le nom de votre fichier de base de données
# Le chemin complet pour la base de données
DATABASE_URL = f"sqlite:///{DB_FILE}"

# --- 2. Charger votre DataFrame existant ---
# Remplacez ceci par le chemin et le nom de votre fichier de données
# Par exemple, si vous avez 'telco_customer_churn.csv' à la racine de votre projet
try:
    df_existing = joblib.load('df_telco_customer_churn.pkl')
    print("DataFrame 'telco_customer_churn.csv' chargé avec succès.")
except FileNotFoundError:
    print(f"Erreur : Le fichier 'telco_customer_churn.csv' n'a pas été trouvé.")
    print("Veuillez vérifier le chemin ou créer un DataFrame de test.")
    

# --- 3. Créer l'engine SQLAlchemy pour la connexion à la base de données ---
# SQLAlchemy permet à Pandas de communiquer avec la base de données
engine = create_engine(DATABASE_URL)

# --- 4. Spécifier le nom de la table dans la base de données ---
TABLE_NAME = "customers" # Le nom de la table où vous voulez insérer vos données

# --- 5. Insérer le DataFrame dans la base de données SQLite ---
try:
    # 'if_exists=' options:
    # 'fail': (défaut) Lance une erreur si la table existe déjà.
    # 'replace': Supprime la table si elle existe, puis la recrée et insère les données.
    # 'append': Insère les données à la suite des données existantes dans la table.
    # Pour la première exécution, 'replace' est souvent pratique pour s'assurer d'une table propre.
    # Pour des ajouts, utilisez 'append'.
    df_existing.to_sql(TABLE_NAME, con=engine, if_exists='replace', index=False)
    print(f"DataFrame inséré avec succès dans la table '{TABLE_NAME}' de '{DB_FILE}'.")

    # --- Optionnel : Vérifier le contenu de la table (pour confirmation) ---
    conn = engine.connect()
    df_from_db = pd.read_sql_table(TABLE_NAME, con=conn)
    conn.close()
    print(f"\nDonnées lues depuis la table '{TABLE_NAME}' :\n", df_from_db.head())

except Exception as e:
    print(f"Une erreur est survenue lors de l'insertion : {e}")