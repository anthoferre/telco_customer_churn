# Jeu de données sur le taux de désabonnement des clients de télécommunications

---

## Technologies et Compétences Clés

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![Bash](https://img.shields.io/badge/Bash-121011?style=for-the-badge&logo=gnu-bash&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

---

## Table des Matières

* [1. Description Générale](#1-description-générale)
* [2. Objectifs du Projet](#2-objectifs-du-projet)
* [3. Contenu du Jeu de Données](#3-contenu-du-jeu-de-données)
* [4. Performance du modèle](#4-performance-du-modele)
* [5. Architecture de la Solution](#5-architecture-de-la-solution)
* [6. Comment Lancer le Projet](#6-comment-lancer-le-projet)
* [7. Utilisation de l'API](#7-utilisation-de-lapi)
    * [7.1. Prédiction du Churn](#71-prédiction-du-churn)
    * [7.2. Ajout de Nouvelles Données Client](#72-ajout-de-nouvelles-données-client)
* [8. Utilisation de l'Application Streamlit](#8-utilisation-de-lapplication-streamlit)
* [9. Ré-entraînement du Modèle](#9-ré-entrainement-du-modele)
* [10. Cas d'Utilisation et Applications Potentielles](#10-cas-dutilisation-et-applications-potentielles)
* [11. Références](#11-références)

---

## 1. Description Générale 

Ce jeu de données fournit un aperçu complet des clients d'une entreprise de télécommunications, en se concentrant sur les facteurs qui influencent leur décision de quitter le service (désabonnement ou "churn"). Il contient des informations démographiques, des détails sur leurs comptes, les services qu'ils utilisent et leurs interactions avec l'entreprise. L'objectif principal de ce jeu de données est de permettre l'analyse du comportement des clients et la prédiction du taux de désabonnement, afin de développer des stratégies de fidélisation efficaces.

Le jeu de données comprend **7 043 lignes**, représentant chacune un client unique, et **21 colonnes**, décrivant diverses caractéristiques de ces clients. La variable cible est la colonne `Churn`, qui indique si le client a quitté le service ou non.

## 2. Objectifs du Projet

- Anticiper le désabonnement : Développer un modèle de Machine Learning capable de prédire avec précision la probabilité qu'un client se désabonne.
- Identifier les facteurs clés : Déterminer quelles caractéristiques (services, contrat, charges, etc.) ont l'impact le plus significatif sur la décision de désabonnement.
- Déployer une solution opérationnelle : Mettre en place une API robuste permettant d'intégrer facilement la prédiction de churn dans les systèmes existants de l'entreprise.
- Permettre l'amélioration continue : Concevoir une architecture facilitant le ré-entraînement du modèle avec de nouvelles données, assurant ainsi la pertinence et la performance du modèle dans le temps (approche MLOps).
   
## 3. Contenu du jeu de données

Les colonnes du jeu de données peuvent être regroupées dans les catégories suivantes :

* **Informations sur le client :**
    * `CustomerID` : Identifiant unique du client.
    * `Gender` : Sexe du client (homme ou femme).
    * `SeniorCitizen` : Indique si le client est une personne âgée (1 pour oui, 0 pour non).
    * `Partner` : Indique si le client a un partenaire (Yes ou No).
    * `Dependents` : Indique si le client a des personnes à charge (Yes ou No).

* **Informations sur le compte du client :**
    * `tenure` : Nombre de mois pendant lesquels le client est resté avec l'entreprise.
    * `Contract` : Type de contrat du client (Month-to-month, One year, Two year).
    * `PaperlessBilling` : Indique si le client a opté pour la facturation électronique (Yes ou No).
    * `PaymentMethod` : Mode de paiement du client (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
    * `MonthlyCharges` : Montant facturé mensuellement au client.
    * `TotalCharges` : Montant total facturé au client jusqu'à présent.

* **Services souscrits par le client :**
    * `PhoneService` : Indique si le client a un service téléphonique (Yes ou No).
    * `MultipleLines` : Indique si le client a plusieurs lignes téléphoniques (Yes, No, No phone service).
    * `InternetService` : Type de service Internet du client (DSL, Fiber optic, No).
    * `OnlineSecurity` : Indique si le client a une sécurité en ligne (Yes, No, No internet service).
    * `OnlineBackup` : Indique si le client a une sauvegarde en ligne (Yes, No, No internet service).
    * `DeviceProtection` : Indique si le client a une protection de l'appareil (Yes, No, No internet service).
    * `TechSupport` : Indique si le client bénéficie d'une assistance technique (Yes, No, No internet service).
    * `StreamingTV` : Indique si le client regarde la télévision en streaming (Yes, No, No internet service).
    * `StreamingMovies` : Indique si le client regarde des films en streaming (Yes, No, No internet service).

* **Variable cible :**
    * `Churn` : Indique si le client s'est désabonné au cours du dernier mois (Yes ou No).

## 4. Performance du modèle

Le modèle de classification (`RandomForestClassifier`) a été évalué sur un jeu de données de test dédié pour garantir la généralisation de ses prédictions.

* **AUC-ROC :** `0.84`

### Matrice de Confusion

|      |Réel |   0 |   1 |   Total |
|------|-----|-----|-----|---------|
|Prédit|-----|-----|-----|---------|
| 0    |-----| 576 | 457 |    1033 |
| 1    |-----|  35 | 339 |     374 |
|Total |-----| 611 | 796 |    1407 | 

### Rapport de Classification

| Métrique    | Precision | Recall | F1-Score | Moy.Geo |
|-------------|-----------|--------|----------|---------|
| Non-Churn   | `0.94`    | `0.56` | `0.70`   | `0.71`  |
| Churn       | `0.43`    | `0.91` | `0.58`   | `0.71`  |
| Moy. pond.  | `0.81`    | `0.65` | `0.67`   | `0.71`  |

*Note : Les valeurs de Precision, Recall et F1-Score pour la classe 'Churn' sont particulièrement importantes pour notre objectif de rétention client. La Moyenne Géommétrique est une métrique importante lors de jeux de données déséquilibrées comme ici lors de l'étude de désabonnement de clients. *

## 5. Architecture de la Solution

```
└── telco_customer_churn\
   ├── .venv/
   ├── README.md
   ├── requirements.txt
   ├── raw_data.csv
   ├── train_model.py
   ├── best_overall_churn_model.pkl
   ├── sql_app.db
   ├── my_streamlit.py
   ├── my_api.py
   └── images/
```

   #### 5.1. API de Prédiction (FastAPI) :

Un service robuste et performant (`my_api.py`) permettant de soumettre les caractéristiques d'un client et d'obtenir en retour une prédiction de désabonnement (probabilité et label).

   #### 5.2. Modèle de Machine Learning (Scikit-learn Pipeline / Joblib) :
Un script Python dédié (`train_model.py`) est responsable de l'entraînement du modèle. Il lit les dernières données depuis la base de données, met à jour la pipeline de ML et sauvegarde la nouvelle version du modèle (`best_overall_churn_model.pkl`).
Le modèle de prédiction est une pipeline Scikit-learn sérialisée avec `joblib`, englobant les étapes de prétraitement des données (encodage des variables catégorielles, standardisation, etc.) et l'algorithme de classification entraîné.
Cette séparation assure que le processus de ré-entraînement n'affecte pas la disponibilité de l'API de prédiction et permet des mises à jour régulières du modèle avec de nouvelles données, essentielle pour maintenir sa performance.

   #### 5.3. Application de Visualisation (Streamlit) :
Une interface utilisateur interactive (`my_streamlit.py`) conçue pour l'analyse graphique du jeu de données, l'étude des relations entre les variables mais aussi elle peut servir de démonstration des prédictions de churn.

   #### 5.4. Base de Données (SQLite / SQLModel) :
Une base de données SQLite (`sql_app.db`) est utilisée pour stocker les données historiques des clients (y compris le statut de churn réel) ainsi que potentiellement les logs des prédictions effectuées.
SQLModel est utilisé pour une interaction ORM (Object-Relational Mapping) simple et efficace avec la base de données.



## 6. Comment Lancer le Projet

   #### 6.1. Cloner le dépôt :

```Bash
git clone https://github.com/anthoferre/telco_customer_churn.git
cd telco_customer_churn
```


   #### 6.2. Créer un environnement virtuel (recommandé) :
```Bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .\.venv\Scripts\activate
```

   #### 6.3. Installer les dépendances :
```Bash
pip install -r requirements.txt
```

   #### 6.4. Pré-requis Modèle : Assurez-vous d'avoir un fichier `churn_model.pkl` (le modèle de ML pré-entraîné) dans le répertoire racine du projet. Si vous n'en avez pas, vous devrez d'abord exécuter le script de ré-entraînement.

   #### 6.5. Lancer l'API :
```Bash
uvicorn my_api:api --reload
```
L'API sera accessible à l'adresse `http://127.0.0.1:8000`. La documentation interactive (Swagger UI) est disponible à `http://127.0.0.1:8000/docs`.

## 7. Utilisation de l'API

   #### 7.1. Prédiction du Churn

Endpoint : `POST /predict_churn`
Description : Prédit la probabilité de désabonnement d'un client en fonction de ses caractéristiques.

Exemple de Requête (`curl`) :
```Bash

    curl -X 'POST' \
      'http://127.0.0.1:8000/predict_churn' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "Gender": "Male",
        "Seniorcitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure": 24,
        "Phoneservice": "Yes",
        "Multiplelines": "No",
        "Internetservice": "DSL",
        "Onlinesecurity": "Yes",
        "Onlinebackup": "No",
        "Deviceprotection": "Yes",
        "Techsupport": "Yes",
        "Streamingtv": "No",
        "Streamingmovies": "No",
        "Contract": "One year",
        "Paperlessbilling": "Yes",
        "Paymentmethod": "Mailed check",
        "Monthlycharges": 75.25,
        "Totalcharges": 1806.05
      }'
```
   #### 7.2. Ajout de Nouvelles Données Client

Endpoint : `POST /add_new_customer_data`
Description : Permet d'ajouter de nouvelles données client, y compris leur statut de désabonnement réel, à la base de données. Ces données serviront pour le ré-entraînement futur du modèle.
Exemple de Requête (`curl`) :
```Bash

    curl -X 'POST' \
      'http://127.0.0.1:8000/add_new_customer_data' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "Gender": "Female",
        "Seniorcitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "Tenure": 12,
        "Phoneservice": "Yes",
        "Multiplelines": "Yes",
        "Internetservice": "Fiber optic",
        "Onlinesecurity": "No",
        "Onlinebackup": "Yes",
        "Deviceprotection": "No",
        "Techsupport": "No",
        "Streamingtv": "Yes",
        "Streamingmovies": "Yes",
        "Contract": "Month-to-month",
        "Paperlessbilling": "Yes",
        "Paymentmethod": "Electronic check",
        "Monthlycharges": 95.00,
        "Totalcharges": 1140.00,
        "Churn": "Yes"
      }'
```

## 8. Utilisation de l'Application Streamlit

L'application Streamlit (`my_streamlit.py`) fournit une interface utilisateur graphique pour interagir avec le modèle de prédiction de churn. Elle permet de visualiser l'interface de saisie des données client et d'obtenir des prédictions de manière interactive.

Pour lancer l'application Streamlit :

- Dans votre terminal, naviguez vers le dossier où se trouve `my_streamlit.py`.
- Exécutez la commande suivante :
```Bash
streamlit run my_streamlit.py
```
L'application s'ouvrira automatiquement dans votre navigateur web à l'adresse `http://localhost:8501`.

## 9. Ré-entraînement du Modèle

Le modèle peut être ré-entraîné en exécutant le script dédié :
```Bash

python train_model.py
```
Ce script se connectera à la base de données, récupérera toutes les données disponibles (y compris celles ajoutées via l'API `/add_new_customer_data`), ré-entraînera la pipeline de ML et sauvegardera le nouveau modèle dans `best_overall_churn_model.pkl`. Après le ré-entraînement, l'API FastAPI doit être redémarrée pour charger la nouvelle version du modèle.

## 10. Cas d'utilisation et applications potentielles

Ce jeu de données offre de nombreuses possibilités d'analyse et d'application, notamment :

* **Analyse exploratoire des données (EDA) :** Comprendre la distribution des différentes caractéristiques, identifier les relations entre elles et visualiser les différences entre les clients qui se sont désabonnés et ceux qui sont restés.
* **Modélisation prédictive du taux de désabonnement :** Construire des modèles de classification (par exemple, régression logistique, arbres de décision, forêts aléatoires, réseaux neuronaux) pour prédire la probabilité qu'un client se désabonne.
* **Identification des facteurs clés de désabonnement :** Déterminer les variables qui ont le plus d'influence sur la décision d'un client de quitter le service.
* **Segmentation de la clientèle :** Regrouper les clients en fonction de leurs caractéristiques et de leur comportement pour adapter les stratégies de fidélisation.
* **Développement de stratégies de rétention ciblées :** Concevoir des interventions spécifiques pour les clients à haut risque de désabonnement.
* **Évaluation de l'impact des offres et des promotions :** Analyser si certaines offres ou promotions ont un effet sur la réduction du taux de désabonnement.

 
## 11. Références

[Lien vers le jeu de données sur Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
