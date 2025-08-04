# Importer toutes les librairies
import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.metrics import confusion_matrix, classification_report, recall_score, f1_score

# Configuration de Seaborn
sns.set_theme(style="whitegrid") # Utiliser un thème plus léger et souvent plus rapide à rendre

st.set_page_config(layout="wide", page_title = 'Customer_Churn')

# Masquer le menu de Streamlit et le bouton de déploiement
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Fonctions de chargement des données avec cache ---

# Utilisation de st.cache_data pour les DataFrames qui ne changent pas
@st.cache_data
def load_dataframe(filepath):
    """Charge un DataFrame à partir d'un fichier CSV ou Pickle."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.pkl'):
        return pd.read_pickle(filepath)
    else:
        raise ValueError("Format de fichier non pris en charge. Utilisez .csv ou .pkl")

# Charger les DataFrames une seule fois
df = load_dataframe('../database/raw_data.csv')
df_cleaned = load_dataframe('../database/cleaned_data.pkl')

# --- Préparation des listes de colonnes (peut être mis en cache si df_cleaned est statique) ---
@st.cache_data
def get_cleaned_features_lists(dataframe):
    """Prépare les listes de variables numériques et catégorielles."""
    features_num = [col for col in dataframe.select_dtypes(['int','float']) if dataframe.select_dtypes(['int','float'])[col].nunique() > 2]
    features_cat = [col for col in dataframe.columns if dataframe[col].nunique() <= 10]
    return features_num, features_cat

features_list_df_cleaned_num, features_list_df_cleaned_cat = get_cleaned_features_lists(df_cleaned)

# Fonction pour charger le modèle (avec st.cache_resource pour les objets non-sérialisables comme les modèles)
@st.cache_resource
def load_ml_model(model_path):
    return joblib.load(model_path)

@st.cache_data
def load_test_data(data_path):
    return joblib.load(data_path)

try:
    x_test = load_test_data('../train_model/x_test.pkl')
    y_test = load_test_data('../train_model/y_test.pkl')
except FileNotFoundError:
    st.error("Les fichiers `x_test` ou `y_test` sont introuvables. Assurez-vous qu'ils sont dans le même répertoire que votre script.")
    st.stop()

# # Charger les modèles
try:
    churn_model = load_ml_model('../train_model/best_overall_churn_model.pkl')
except FileNotFoundError:
    st.error("Le fichier du modèle est introuvable. Veuillez vérifier le chemin.")
    st.stop()

# --- Barre latérale pour la navigation ---
choix_partie = st.sidebar.radio(
    "Sommaire",
    [
        "I - Introduction",
        "II - Exploration des données",
        "III - Data visualization",
        "IV - Le meilleur modèle de prédiction du taux de désabonnement des clients",
        'V - Etude de cas',
        "VI - Conclusion et Perspectives"
    ]
)

# --- Contenu principal de l'application ---

# Partie 1 : Introduction
if choix_partie == 'I - Introduction':
    st.title("Taux de désabonnements des clients de télécommunication")
    st.subheader('I - Introduction')
    st.image('https://vertone.com/wp-content/uploads/2018/12/adobestock_436800241-scaled.jpeg', use_container_width=False)
    st.markdown("""
    Ce jeu de données fournit un aperçu complet des clients d'une entreprise de télécommunications, en se concentrant sur les facteurs qui influencent leur décision de quitter le service (désabonnement ou "churn"). Il contient des informations démographiques, des détails sur leurs comptes, les services qu'ils utilisent et leurs interactions avec l'entreprise. 
    \n\nL'objectif principal de ce jeu de données est de permettre l'analyse du comportement des clients et la prédiction du taux de désabonnement, afin de développer des stratégies de fidélisation efficaces.
    """)

# Partie 2 : Exploration des données
elif choix_partie == 'II - Exploration des données':
    st.title('II - Exploration des données') 
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Afficher les informations détaillées"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

    with col2:
        with st.expander("Afficher les statistiques descriptives"):
            st.dataframe(df.describe())

    st.header("Aperçu Initial du Jeu de Données")
    st.subheader("1. Structure du Jeu de Données")
    st.markdown(f"**Taille :** L'ensemble de données se compose de **{df.shape[0]} lignes** d'informations, réparties sur **{df.shape[1]} colonnes** distinctes.")
    st.markdown("La majorité des colonnes, soit **18**, contiennent des données de type texte (`object`). **2** colonnes contiennent des nombres entiers (`int`). **1** colonne contient des nombres décimaux (`float`).")

    st.subheader("2. Points Requérant une Attention Particulière (Qualité des Données)")
    st.markdown("**- Colonne 'TotalCharges'**")
    st.markdown("""
    Actuellement identifiée comme contenant du texte (`object`), cette colonne devrait normalement contenir des nombres décimaux (`float`) représentant le montant total facturé aux clients. L'examen a révélé que **11 lignes** présentent des valeurs manquantes dans cette colonne, ce qui est probablement la raison de son type de données incorrect.
    \nSolution : supprimer les 11 lignes où TotalCharges est manquant semble être une approche raisonnable pour commencer, étant donné le faible nombre de lignes concernées. Ce qui permettra de convertir la colonne en nombres décimaux.
    """)

    st.markdown("**- Colonnes Binaires**")
    st.markdown("Plusieurs colonnes représentent des choix binaires (par exemple, 'Oui'/'Non', 'Vrai'/'Faux'). Bien qu'actuellement de type texte (`object`), il serait avantageux de les convertir en valeurs numériques (0 et 1) pour faciliter les analyses quantitatives et la modélisation.")

    st.subheader("3. Informations Clés sur les Données")
    st.markdown("**- Absence de Doublons :**")
    st.markdown("L'analyse de la colonne 'customerID' n'a révélé aucune valeur dupliquée, ce qui indique qu'il n'y a pas de lignes complètement identiques dans l'ensemble du jeu de données.")

    st.markdown("**- Déséquilibre de la Variable Cible ('Churner')**")
    st.markdown(f"La variable que nous cherchons à prédire ('Churn', indiquant si un client s'est désabonné ou non) est déséquilibrée. Environ **{np.round(df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)}%** des clients se sont désabonnés, tandis que la majorité est restée. Ce déséquilibre devra être pris en compte lors de la construction de modèles prédictifs (technique de sous ou sur échantillonnage).")

    st.markdown("**- Absence de Valeurs Aberrantes Numériques**")
    st.markdown("Une première vérification des colonnes contenant des nombres n'a pas révélé de valeurs anormalement éloignées des autres, suggérant une certaine cohérence dans les données numériques.")

    st.header("Préprocessing")
    st.code('''# sélectionner les variables binaires Yes or No et les remplacer respectivement par 1 et 0
def selectionner_colonnes_binaires (df):
    colonnes_binaires = []
    for col in df.select_dtypes('object').columns:
        if set(df[col].unique()) == {"Yes","No"}:
            colonnes_binaires.append(col)
    return colonnes_binaires

df[selectionner_colonnes_binaires] = df[selectionner_colonnes_binaires(df)].replace({"Yes" : 1, "No" : 0})

# remplacer female par 0 et male par 1
df['gender'] = df['gender'].replace({"Female" : 0, "Male" : 1})

# one hot encoding
df = pd.get_dummies(data = df)

# supprimer toutes les colonnes contenant no internet service sauf une car toutes identiques
df = df.drop([col for col in df.columns if "No internet service" in col][1:], axis = 1)

#Supprimer la colonne no_phone_service de multiplelines car info déjà dans la variable PhoneService
df.drop('MultipleLines_No phone service', axis = 1, inplace=True)

# Renommer la colonne "No_internet_service"
df = df.rename(columns={"OnlineSecurity_No internet service" : "No_internet_service"})''')

# Partie 3 : Data Visualization
elif choix_partie == 'III - Data visualization':
    st.title('III - Data visualization')

    type_graphique = [
        "Distribution d'une variable",
        "Distribution des variables numériques par statut de désabonnement",
        "Taux de désabonnement par variable catégorielle",
        "Proportion",
        "Corrélation"
    ]
    
    col1, col2 = st.columns([1.5,2.5])
    with col1:
        graphique_choisi = st.radio("Quel est le type d'analyse graphique souhaitée?", type_graphique)
    
    with col2:
        if graphique_choisi == "Distribution d'une variable":

            x_choisi = st.selectbox(label='Choisir une variable en abscisse', options=df_cleaned.columns)
        
            fig, ax = plt.subplots(figsize=(10, 6))
            if df_cleaned[x_choisi].nunique() > 5: # Utiliser nunique pour distinguer numérique/catégorielle avec plusieurs valeurs
                sns.histplot(df_cleaned[x_choisi], stat='percent', ax=ax)
                ax.set_title(f'Distribution de la variable {x_choisi}', fontsize=16)
                ax.set_ylabel('Pourcentage')
            else:
                sns.countplot(x=df_cleaned[x_choisi], hue=df_cleaned[x_choisi], stat='percent', palette='viridis', ax=ax, legend=False)
                ax.set_title(f'Distribution de la variable {x_choisi}', fontsize=16)
                ax.set_ylabel('Pourcentage')
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)
            st.pyplot(fig)
            plt.close(fig) # Fermer la figure pour libérer de la mémoire

        if graphique_choisi == 'Distribution des variables numériques par statut de désabonnement':
            x_choisi = st.selectbox(label='Choisir une variable en abscisse', options=features_list_df_cleaned_num)
            
            fig, axes = plt.subplots(1, 2, figsize=(18, 7)) # Agrandir légèrement pour la clarté

            sns.histplot(data=df_cleaned, x=x_choisi, hue='Churn', multiple='fill', bins=20, ax=axes[0], palette='coolwarm')
            axes[0].set_title(f'Distribution du Churn par {x_choisi}', fontsize=14)
            axes[0].set_xlabel(f'{x_choisi}')
            axes[0].set_ylabel('Proportion de Churn')
            axes[0].set_yticks(np.arange(0, 1.1, 0.2))
            axes[0].set_yticklabels([f'{int(p*100)}%' for p in np.arange(0, 1.1, 0.2)])
            axes[0].legend(title='Churn', labels=['Non', 'Oui']) # S'assurer que les labels de légende sont clairs

            sns.boxplot(data=df_cleaned, y=x_choisi, hue='Churn', ax=axes[1], palette='Set2')
            axes[1].set_title(f'Distribution de {x_choisi} par Statut de Churn', fontsize=14)
            axes[1].set_xlabel('Statut de Churn')
            axes[1].set_ylabel(f'{x_choisi}')
            axes[1].set_xticks([0, 1])
            axes[1].set_xticklabels(['Non-Churn', 'Churn']) # Labels plus descriptifs
            
            plt.tight_layout() # Ajuste automatiquement les paramètres des sous-figures pour qu'elles rentrent dans la zone de la figure.
            st.pyplot(fig)
            plt.close(fig)

        elif graphique_choisi == 'Taux de désabonnement par variable catégorielle':
            x_choisi = st.selectbox(label='Choisir une variable d\'intérêt', options=features_list_df_cleaned_cat)

            fig, ax = plt.subplots(figsize=(12, 7))

            df_crosstab = pd.crosstab(df_cleaned[x_choisi], df_cleaned.Churn, normalize='index') * 100
            df_crosstab = df_crosstab.reset_index().melt(id_vars = x_choisi, var_name = 'Churn', value_name = 'Percentage')
            sns.barplot(data = df_crosstab, x = x_choisi, y = 'Percentage', hue = 'Churn', ax = ax) 
            ax.set_title(f"Taux de désabonnement en fonction de la variable {x_choisi}")
            # Ajout des pourcentages sur les barres pour une lecture facile
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)

            # Vérifier si les valeurs uniques sont 0 et 1 pour les changer en "Non" et "Oui"
            if set(df_cleaned[x_choisi].unique()) == {0, 1}:
                ax.set_xticks([0, 1]) # Fixe les positions des ticks
                ax.set_xticklabels(['Non', 'Oui']) # Applique les nouvelles étiquettes
            elif set(df_cleaned[x_choisi].unique()) == {1, 0}: # Au cas où l'ordre serait inversé
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Oui', 'Non']) # Ajuster selon l'ordre réel des catégories si nécessaire

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            

        elif graphique_choisi == 'Proportion':
            x_choisi = st.selectbox(label='Choisir une variable d\'intérêt', options=features_list_df_cleaned_cat)
            
            fig, ax = plt.subplots(figsize=(2, 2)) # Agrandir légèrement pour meilleure lisibilité
            plt.title(f'Proportion de la variable {x_choisi}', fontsize=16)
            
            # Obtenir les counts et les index (labels)
            counts = df_cleaned[x_choisi].value_counts()
            labels = counts.index
            
            # Utiliser autopct='%1.1f%%' directement dans pie
            ax.pie(x=counts, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 7}, pctdistance=0.7)
            ax.axis('equal')  # Assure que le cercle est parfait
            
            st.pyplot(fig)
            plt.close(fig)

        elif graphique_choisi == 'Corrélation':
            fig, ax = plt.subplots(figsize=(7, 5)) # Agrandir la heatmap pour plus de lisibilité
            sns.heatmap(data=df_cleaned.select_dtypes(['int', 'float']).corr(), annot=True, fmt='.2f', annot_kws={'size': 9}, cmap='coolwarm', ax=ax)
            plt.title('Corrélation entre les différentes variables du jeu de données', fontdict={'fontsize': 18})
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# Partie 4 : Modèle de prédiction du taux de désabonnement des clients
elif choix_partie == 'IV - Le meilleur modèle de prédiction du taux de désabonnement des clients':
    st.title('IV - Le meilleur modèle de prédiction du taux de désabonnement des clients')

    def plot_confusion_matrix(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5)) # Taille fixe pour la matrice
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Prédictions')
        ax.set_ylabel('Vraies valeurs')
        ax.set_title('Matrice de Confusion')
        return fig
    
    def display_model_info(title, model, X_test_data, y_test_data, labels_data):
        y_pred = model.predict(X_test_data)
        recall = recall_score(y_test_data, y_pred)
        f1 = f1_score(y_test_data, y_pred)

        st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 1.2em'>{title}</div>", unsafe_allow_html=True)

        y_pred = model.predict(X_test_data)

        cm_fig = plot_confusion_matrix(y_test_data, y_pred, labels_data)
        st.pyplot(cm_fig)
        plt.close(cm_fig) # Fermer la figure

        st.metric(label = 'F1-Score', value = np.round(f1,3), border = True)
        st.metric(label = 'Recall-Score (Churners)', value = np.round(recall,3), border = True)
        

   
    labels_churn = [0, 1] # Assurez-vous que c'est cohérent avec vos données

    display_model_info('Meilleur modèle', churn_model, x_test, y_test, labels_churn)

   

# Partie 5 : Etude de cas sur de nouvelles données
elif choix_partie == 'V - Etude de cas':
    st.title('V - Etude de cas')
        # --- 3. Définir les features et leurs plages/options ---
    # Ces informations doivent correspondre aux features utilisées lors de l'entraînement du modèle
    features_info = {
        'Tenure': {'type': 'slider', 'min': 0, 'max': 72, 'default': 30, 'step': 1, 'label': 'Ancienneté (mois)'},
        'MonthlyCharges': {'type': 'slider', 'min': 18.0, 'max': 120.0, 'default': 50.0, 'step': 0.5, 'label': 'Frais Mensuels (€)'},
        'TotalCharges': {'type': 'number_input', 'min': 0.0, 'max': 9000.0, 'default': 1500.0, 'step': 10.0, 'label': 'Frais Totaux (€)'},
        'Gender': {'type': 'radio', 'options': ['Male', 'Female'], 'default': 'Male', 'label': 'Genre'},
        'InternetService': {'type': 'selectbox', 'options': ['DSL', 'Fiber optic', 'No'], 'default': 'Fiber optic', 'label': 'Service Internet'},
        'Contract': {'type': 'selectbox', 'options': ['Month-to-month', 'One year', 'Two year'], 'default': 'Month-to-month', 'label': 'Contrat'},
        'Partner': {'type': 'radio', 'options': ['Yes', 'No'], 'default': 'No', 'label': 'Partenaire'},
        'Dependents': {'type': 'radio', 'options': ['Yes', 'No'], 'default': 'No', 'label': 'Dépendants'},
        # Ajoute d'autres features pertinentes de ton modèle
    }

    # --- 4. Créer les widgets dans la barre latérale ---
    st.sidebar.header("Paramètres du Client Hypothétique")
    input_data = {}

    for feature, info in features_info.items():
        if info['type'] == 'slider':
            input_data[feature] = st.sidebar.slider(
                label=info['label'],
                min_value=info['min'],
                max_value=info['max'],
                value=info['default'],
                step=info['step']
            )
        elif info['type'] == 'number_input':
            input_data[feature] = st.sidebar.number_input(
                label=info['label'],
                min_value=info['min'],
                max_value=info['max'],
                value=info['default'],
                step=info['step']
            )
        elif info['type'] == 'selectbox':
            input_data[feature] = st.sidebar.selectbox(
                label=info['label'],
                options=info['options'],
                index=info['options'].index(info['default']) # Pour que la valeur par défaut soit sélectionnée
            )
        elif info['type'] == 'radio':
            input_data[feature] = st.sidebar.radio(
                label=info['label'],
                options=info['options'],
                index=info['options'].index(info['default'])
            )

    all_model_features = [col for col in df_cleaned.columns if (col != 'Churn' and col != 'Charges_rate')]

    for feature in all_model_features:
        if feature not in input_data: # Si la feature n'a pas été définie par un widget
            # Calculer le mode de cette colonne depuis le DataFrame original
            # .mode()[0] est utilisé car .mode() peut renvoyer plusieurs modes s'il y a des ex-aequo
            mode_value = df_cleaned[feature].mode()[0]
            input_data[feature] = mode_value

    # --- 5. Construire le DataFrame pour la prédiction ---
    # Créer un DataFrame avec une seule ligne à partir des inputs de l'utilisateur
    # Il est crucial que les noms de colonnes correspondent exactement à ceux utilisés lors de l'entraînement
    input_df = pd.DataFrame([input_data])

    # --- 6. Faire la prédiction ---
    if st.button("Prédire le Désabonnement"): # Ajout d'un bouton pour déclencher la prédiction
        st.subheader("Résultat de la Prédiction")
        # Utilisation de try-except pour attraper d'éventuelles erreurs de prédiction
        try:
            # predict_proba renvoie les probabilités pour chaque classe (non-churn, churn)
            prediction_proba = churn_model.predict_proba(input_df)[0]
            # predict renvoie la classe prédite (0 ou 1)
            prediction_class = churn_model.predict(input_df)[0]

            churn_probability = prediction_proba[1] # Probabilité de la classe 1 (churn)
            non_churn_probability = prediction_proba[0] # Probabilité de la classe 0 (non-churn)

            st.write(f"Probabilité de **Non-Désabonnement** : **`{non_churn_probability:.2%}`**")
            st.write(f"Probabilité de **Désabonnement (Churn)** : **`{churn_probability:.2%}`**")

            if prediction_class == 1:
                st.error("Ce client hypothétique est **susceptible de se désabonner**.")
            else:
                st.success("Ce client hypothétique est **peu susceptible de se désabonner**.")

            st.markdown("---")
            st.write("Détails de l'instance client générée :")
            st.dataframe(input_df)

        except Exception as e:
            st.error(f"Une erreur est survenue lors de la prédiction : {e}")
            st.write("Veuillez vérifier que les types de données et les noms de colonnes correspondent à ceux attendus par le modèle.")

# Partie 6 : Conclusion et Perspectives
elif choix_partie == 'VI - Conclusion et Perspectives':
    st.title('VI - Conclusion et Perspectives')

    st.markdown("---")
    st.markdown("#### Conclusion")
    st.markdown("""
    Cette exploration et modélisation du taux de désabonnement des clients de télécommunication a permis de mettre en lumière plusieurs aspects cruciaux. L'analyse des données brutes a révélé des défis initiaux concernant le type de certaines variables et la présence de quelques valeurs manquantes, qui ont été adressés par un pré-traitement ciblé.

    La visualisation des données a ensuite offert des insights précieux sur la distribution des différentes caractéristiques et leur relation avec la variable cible 'Churn'. Des tendances claires se sont dégagées, suggérant que certains facteurs comme la durée d'abonnement, les charges mensuelles et totales, ainsi que l'utilisation de certains services (comme le streaming de films) sont fortement corrélés au désabonnement.

    La comparaison de différents modèles de classification (Régression Logistique, Support Vector Machine, Arbre de Décision et Forêt Aléatoire) a permis d'évaluer leur performance dans la prédiction du 'Churn'. L'examen des matrices de confusion, des rapports de classification et des hyperparamètres optimisés fournit une base solide pour choisir le modèle le plus adapté aux objectifs de l'entreprise. Il est notable que certaines variables, telles que la présence de l'option 'contract_month_to_month', la durée d'abonnement ('tenure') et les charges ('totalcharges' et 'monthly charges'), se sont avérées être des indicateurs clés dans les décisions des modèles.
    """)

    st.markdown("---")
    st.markdown("#### Perspectives")
    st.markdown("""
    Forts de ces résultats, plusieurs pistes d'action et de développement peuvent être envisagées :

    **1. Stratégies de Rétention Ciblées :**
    - Les insights tirés de la visualisation et de l'importance des variables peuvent être utilisés pour identifier les clients à haut risque de désabonnement. Des offres personnalisées ou des interventions proactives pourraient être mises en place pour les fidéliser.
    - Une attention particulière devrait être portée aux clients ayant des durées d'abonnement courtes, des charges mensuelles élevées, un abonnement payé mois par mois (sans engagement) ou qui n'utilisent pas certains services spécifiques.

    **2. Amélioration des Services et de l'Expérience Client :**
    - Comprendre pourquoi les clients qui utilisent certains services sont plus susceptibles de rester (ou de partir) peut orienter l'amélioration de ces services ou l'offre de services complémentaires pertinents.
    - L'analyse des commentaires clients (si disponibles dans d'autres sources de données) pourrait enrichir la compréhension des raisons du désabonnement au-delà des variables quantitatives.

    **3. Optimisation des Modèles de Prédiction :**
    - L'exploration de techniques d'équilibrage des classes plus avancées pourrait potentiellement améliorer la performance des modèles, en particulier pour la détection des clients qui vont se désabonner.
    - L'intégration de nouvelles variables (par exemple, les interactions du service client, les données de navigation sur le site web) pourrait enrichir le pouvoir prédictif des modèles.
    - Un suivi régulier des performances des modèles en production et un réentraînement périodique sont essentiels pour maintenir leur précision dans le temps.

    **4. Développement d'Outils d'Aide à la Décision :**
    - L'application Streamlit elle-même pourrait être transformée en un outil interactif pour les équipes marketing et commerciales, leur permettant de visualiser les risques de désabonnement pour des segments de clients spécifiques et de planifier des actions en conséquence.

    **5. Exploration de Modèles Plus Avancés :**
    - L'expérimentation avec des modèles de machine learning plus complexes (par exemple, réseaux de neurones, méthodes d'ensemble avancées) pourrait être envisagée pour tenter d'améliorer encore la précision des prédictions.

    En conclusion, cette analyse fournit une base solide pour comprendre et potentiellement réduire le taux de désabonnement des clients. En mettant en œuvre des stratégies basées sur ces insights et en continuant à affiner les modèles de prédiction, l'entreprise de télécommunications peut améliorer significativement la fidélisation de sa clientèle et optimiser ses opérations commerciales.
    """)